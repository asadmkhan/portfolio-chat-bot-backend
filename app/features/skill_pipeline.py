from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.normalized import NormalizedJD, NormalizedResume
from app.taxonomy import TaxonomyProvider

from .analysis_units import AnalysisUnit

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]{1,}")
_SOFT_HINTS = {
    "detail-oriented",
    "detail oriented",
    "communication",
    "independent",
    "deadline",
    "deadlines",
    "collaboration",
    "teamwork",
    "adaptability",
    "ownership",
    "profile",
    "professional",
    "resume",
    "access",
    "comfort",
}
_GENERIC_STOP = {
    "resume",
    "access",
    "comfort",
    "profile",
    "detail",
    "ability",
    "professional",
    "tools",
    "experience",
    "tasks",
    "requirements",
    "responsibilities",
    "work",
    "role",
    "team",
    "skills",
    "prior",
    "strong",
    "familiarity",
    "capable",
    "following",
    "precise",
    "instructions",
    "comfortable",
    "working",
    "independently",
    "meeting",
    "tight",
    "deadlines",
    "physical",
    "fresh",
    "user",
    "required",
    "data",
    "collection",
    "annotation",
    "recording",
    "documenting",
    "workflows",
    "custom",
    "follow",
    "staging",
    "record",
    "sessions",
}
_KNOWN_HARD = {
    "windows",
    "macos",
    "linux",
    "sql",
    "python",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "screen recording",
    "record screen sessions",
    "annotate screenshots",
    "bounding boxes",
    "capture tool",
    "staging instructions",
    "qa",
    "quality assurance",
    "macintosh",
    "adobe",
    "adobe acrobat",
    "photoshop",
}


@dataclass(slots=True)
class SkillAlignmentResult:
    denominator: int
    jd_hard_terms: list[str] = field(default_factory=list)
    matched_hard_terms: list[str] = field(default_factory=list)
    missing_hard_terms: list[str] = field(default_factory=list)
    used_llm: bool = False
    llm_fallback: bool = False


def _tokens(text: str) -> list[str]:
    output: list[str] = []
    for token in _WORD_RE.findall(text or ""):
        cleaned = token.lower().strip(".,;:()[]{}")
        if cleaned:
            output.append(cleaned)
    return output


def _normalize_phrase(phrase: str) -> str:
    return re.sub(r"\s+", " ", (phrase or "").strip().lower())


def _extract_candidate_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    clean = _normalize_phrase(text)
    if not clean:
        return phrases

    for known in _KNOWN_HARD:
        if " " in known and known in clean:
            phrases.add(known)

    # Keep short noun-like chunks split by punctuation.
    for chunk in re.split(r"[;,/()]", clean):
        part = chunk.strip(" -:\t")
        if not part:
            continue
        words = [w for w in _tokens(part) if w not in _GENERIC_STOP]
        while words and words[0] in {"and", "or", "with", "including"}:
            words = words[1:]
        if not words:
            continue
        if len(words) == 1:
            word = words[0]
            if word in _KNOWN_HARD or len(word) >= 4:
                phrases.add(word)
            continue
        # Keep compact phrases only; long fluff phrases are noisy in denominator/evidence.
        if len(words) <= 4:
            phrase = " ".join(words)
            if not (
                phrase not in _KNOWN_HARD
                and any(token in {"including", "software"} for token in words)
            ):
                phrases.add(phrase)
            for word in words:
                if word in _KNOWN_HARD:
                    phrases.add(word)
        else:
            for word in words:
                if word in _KNOWN_HARD:
                    phrases.add(word)
    return phrases


def _is_hard_skill(phrase: str) -> bool:
    lower = _normalize_phrase(phrase)
    if not lower:
        return False
    if lower in _SOFT_HINTS:
        return False
    if any(token in _SOFT_HINTS for token in lower.split()):
        return False
    if lower in _KNOWN_HARD:
        return True
    if any(marker in lower for marker in ("os", "linux", "windows", "macos", "sql", "python", "screen", "screenshot", "qa")):
        return True
    if re.search(r"[+#./]", lower):
        return True
    return False


def _canonical_key(phrase: str, taxonomy_provider: TaxonomyProvider) -> tuple[str, str]:
    normalized, canonical = taxonomy_provider.normalize_skill(phrase)
    normalized = _normalize_phrase(normalized)
    canonical_key = (canonical or "").strip().lower()
    return normalized, canonical_key


def _resume_text_corpus(normalized_resume: NormalizedResume, analysis_units: list[AnalysisUnit] | None) -> str:
    chunks: list[str] = []
    chunks.extend(normalized_resume.skills)
    chunks.extend(claim.text for claim in normalized_resume.claims if claim.text)
    if analysis_units:
        chunks.extend(unit.text for unit in analysis_units if unit.text)
    return "\n".join(chunks).lower()


def build_skill_alignment(
    *,
    normalized_resume: NormalizedResume,
    normalized_jd: NormalizedJD,
    analysis_units: list[AnalysisUnit] | None,
    taxonomy_provider: TaxonomyProvider,
    allow_llm: bool = False,
) -> SkillAlignmentResult:
    # Step-1 deterministic ATS path; LLM flag intentionally ignored.
    _ = allow_llm

    jd_chunks: list[str] = []
    jd_chunks.extend(req.text for req in normalized_jd.requirements if req.text)
    jd_chunks.extend(normalized_jd.responsibilities)
    if normalized_jd.title:
        jd_chunks.append(normalized_jd.title)

    jd_candidates: set[str] = set()
    for chunk in jd_chunks:
        jd_candidates |= _extract_candidate_phrases(chunk)

    hard_terms: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for phrase in sorted(jd_candidates):
        if not _is_hard_skill(phrase):
            continue
        normalized, canonical_key = _canonical_key(phrase, taxonomy_provider)
        key = (normalized, canonical_key)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        hard_terms.append(normalized)

    resume_corpus = _resume_text_corpus(normalized_resume, analysis_units)
    resume_tokens = set(_tokens(resume_corpus))
    resume_norm_map: set[tuple[str, str]] = set()
    for token in sorted(resume_tokens):
        normalized, canonical_key = _canonical_key(token, taxonomy_provider)
        resume_norm_map.add((normalized, canonical_key))

    matched: list[str] = []
    missing: list[str] = []
    for term in hard_terms:
        normalized, canonical_key = _canonical_key(term, taxonomy_provider)
        term_tokens = [token for token in normalized.split() if token not in _GENERIC_STOP]
        token_match = bool(term_tokens) and all(token in resume_corpus for token in term_tokens)
        canonical_match = (normalized, canonical_key) in resume_norm_map or (
            canonical_key and any(key[1] == canonical_key for key in resume_norm_map)
        )
        if token_match or canonical_match:
            matched.append(normalized)
        else:
            missing.append(normalized)

    return SkillAlignmentResult(
        denominator=len(hard_terms),
        jd_hard_terms=hard_terms,
        matched_hard_terms=matched,
        missing_hard_terms=missing,
        used_llm=False,
        llm_fallback=False,
    )
