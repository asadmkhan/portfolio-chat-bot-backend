from __future__ import annotations

import re

from app.core.config.scoring import get_scoring_value
from app.schemas.normalized import MatchHit, MatchMatrix, NormalizedJD, NormalizedResume
from app.taxonomy.local_taxonomy import LocalTaxonomy
from app.taxonomy.provider import TaxonomyProvider

from .embeddings import EmbeddingProvider, SimpleEmbeddingProvider, cosine_similarity

_TOKEN_PATTERN = re.compile(r"[a-z0-9\+#]+")
_METRIC_PATTERN = re.compile(r"(\d[\d,\.]*\s*%|\$\s*\d[\d,\.]*|€\s*\d[\d,\.]*|\b\d+\b)")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _extract_canonical_skill_ids(text: str, taxonomy: TaxonomyProvider) -> set[str]:
    words = _TOKEN_PATTERN.findall(text.lower())
    canonical_ids: set[str] = set()
    for n in (1, 2, 3):
        for idx in range(0, len(words) - n + 1):
            phrase = " ".join(words[idx : idx + n])
            _, canonical_id = taxonomy.normalize_skill(phrase)
            if canonical_id:
                canonical_ids.add(canonical_id)
    return canonical_ids


def _has_metric_signal(text: str) -> bool:
    return bool(_METRIC_PATTERN.search(text))


def _evidence_strength(similarity: float, weak_match: float, strong_match: float, claim_text: str) -> int:
    if similarity < weak_match:
        return 0
    if similarity < strong_match:
        return 1
    if _has_metric_signal(claim_text):
        return 3
    return 2


def build_match_matrix(
    jd: NormalizedJD,
    resume: NormalizedResume,
    top_n: int = 5,
    embedding_provider: EmbeddingProvider | None = None,
    taxonomy_provider: TaxonomyProvider | None = None,
) -> MatchMatrix:
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0")

    weak_match = float(get_scoring_value("matching.similarity_thresholds.weak_match", 0.55))
    strong_match = float(get_scoring_value("matching.similarity_thresholds.strong_match", 0.72))

    embedder = embedding_provider or SimpleEmbeddingProvider()
    taxonomy = taxonomy_provider or LocalTaxonomy()

    requirement_texts = [req.text for req in jd.requirements]
    claim_texts = [claim.text for claim in resume.claims]

    requirement_vectors = embedder.embed(requirement_texts) if requirement_texts else []
    claim_vectors = embedder.embed(claim_texts) if claim_texts else []

    requirement_tokens = [_tokenize(text) for text in requirement_texts]
    claim_tokens = [_tokenize(text) for text in claim_texts]
    requirement_skills = [_extract_canonical_skill_ids(text, taxonomy) for text in requirement_texts]
    claim_skills = [_extract_canonical_skill_ids(text, taxonomy) for text in claim_texts]

    matches: dict[str, list[MatchHit]] = {}
    for req_idx, requirement in enumerate(jd.requirements):
        req_hits: list[MatchHit] = []
        for claim_idx, claim in enumerate(resume.claims):
            token_sim = _jaccard_similarity(requirement_tokens[req_idx], claim_tokens[claim_idx])
            embed_sim = cosine_similarity(requirement_vectors[req_idx], claim_vectors[claim_idx])

            similarity = (0.6 * token_sim) + (0.4 * embed_sim)
            if requirement_skills[req_idx] & claim_skills[claim_idx]:
                similarity = min(1.0, similarity + 0.5)

            req_hits.append(
                MatchHit(
                    claim_id=claim.claim_id,
                    similarity=round(similarity, 4),
                    evidence_strength=_evidence_strength(
                        similarity=similarity,
                        weak_match=weak_match,
                        strong_match=strong_match,
                        claim_text=claim.text,
                    ),
                )
            )

        req_hits.sort(key=lambda item: item.similarity, reverse=True)
        matches[requirement.req_id] = req_hits[:top_n]

    return MatchMatrix(
        matches=matches,
        must_req_ids=[req.req_id for req in jd.requirements if req.priority == "must"],
        nice_req_ids=[req.req_id for req in jd.requirements if req.priority == "nice"],
    )
