from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from app.schemas.recruiter import (
    ATSHumanRiskSplitRequest,
    ATSHumanRiskSplitResponse,
    BiasRiskFlag,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
    HiringBiasRiskDetectorRequest,
    HiringBiasRiskDetectorResponse,
    JDMarketConcern,
    JDMarketRealityRequest,
    JDMarketRealityResponse,
    JDQualityAnalyzerRequest,
    JDQualityIssue,
    JDQualityResponse,
    MustHaveNiceToHaveBreakdown,
    QuickWin,
    ResumeSignalDimension,
    ResumeSignalStrengthRequest,
    ResumeSignalStrengthResponse,
    RecruiterShareCreateRequest,
    RoleSeniorityDefinitionRequest,
    RoleSeniorityDefinitionResponse,
    RoleSenioritySignals,
    ShortlistCandidateNote,
    ShortlistJustificationRequest,
    ShortlistJustificationResponse,
    ResumeAuthenticityRequest,
    ResumeAuthenticityResponse,
    ResumeAuthenticitySignal,
    ResumeCompareCandidateOutput,
    ResumeCompareRequest,
    ResumeCompareResponse,
    RiskBlock,
    RoleLevelInference,
)
from app.services.tools_llm import ToolsLLMError, json_completion, json_completion_required, tools_llm_enabled

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-/]{1,}")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
_METRIC_RE = re.compile(
    r"(\b\d+(?:\.\d+)?\s?(?:%|percent|ms|s|sec|seconds|mins?|minutes|hours?|days?|weeks?|months?|years?|x)\b|\$\s?\d[\d,]*|\b\d[\d,]*(?:\.\d+)?\b)",
    re.IGNORECASE,
)
_ROLE_KEYWORDS: dict[RoleLevelInference, tuple[str, ...]] = {
    "Junior": ("junior", "entry", "associate", "graduate", "intern"),
    "Mid": ("mid", "intermediate", "regular"),
    "Senior": ("senior", "sr.", "experienced"),
    "Lead": ("lead", "principal", "manager"),
    "Staff": ("staff", "architect", "head of"),
    "Unclear": tuple(),
}
_CICHE_PHRASES = (
    "results-driven",
    "team player",
    "hard-working",
    "go-getter",
    "fast-paced",
    "detail-oriented",
    "synergy",
    "dynamic environment",
    "think outside the box",
)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "your",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "will",
    "can",
    "not",
    "you",
    "our",
    "their",
    "job",
    "role",
    "resume",
}

_MUST_HAVE_MARKERS = ("must", "required", "requirement", "need to", "needs to", "mandatory")
_NICE_TO_HAVE_MARKERS = ("nice to have", "preferred", "plus", "bonus", "optional")
_LEADERSHIP_MARKERS = ("lead", "manage", "mentor", "coach", "stakeholder", "cross-functional")
_ARCH_MARKERS = ("architecture", "architect", "system design", "distributed", "scalable", "platform")
_AUTONOMY_MARKERS = ("independent", "ownership", "autonomous", "self-directed", "drive")
_EXECUTION_MARKERS = ("deliver", "ship", "implement", "execute", "maintain", "operate")
_BIAS_PATTERNS: dict[str, tuple[str, ...]] = {
    "gender-coded": ("rockstar", "ninja", "dominant", "competitive warrior"),
    "age-coded": ("young", "digital native", "recent graduate", "energetic young"),
    "culture-fit vague": ("culture fit", "work hard play hard", "fits our vibe"),
    "ableist": ("able-bodied", "must be physically fit", "crazy", "insane"),
    "aggressive tone": ("must dominate", "aggressive hunter", "killer instinct"),
    "elitism/credential bias": ("ivy league", "top-tier school", "elite university"),
    "location bias": ("local candidates only", "must live nearby", "within commuting distance only"),
    "visa bias": ("no sponsorship", "citizens only", "must have unrestricted work authorization"),
}


class RecruiterQualityError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_strict_mode() -> bool:
    raw = (os.getenv("TOOLS_STRICT_LLM") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text or "").strip()


def _split_lines(text: str) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines()]
    return [line for line in lines if line]


def _extract_candidate_claims(text: str, limit: int = 10) -> list[str]:
    claims: list[str] = []
    for line in _split_lines(text):
        if _BULLET_RE.search(line) or any(v in line.lower() for v in ("built", "led", "implemented", "designed", "launched", "improved", "migrated")):
            claims.append(_normalize_whitespace(line))
    if not claims:
        claims = [_normalize_whitespace(s) for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 30]
    return claims[:limit]


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def _top_terms(text: str, limit: int = 25) -> list[str]:
    tokens = [token for token in _tokenize(text) if len(token) > 2 and token not in _STOPWORDS]
    counts = Counter(tokens)
    return [term for term, _ in counts.most_common(limit)]


def _find_snippets(text: str, terms: list[str], limit: int = 3) -> list[str]:
    snippets: list[str] = []
    for line in _split_lines(text):
        lower = line.lower()
        if any(term in lower for term in terms):
            snippets.append(line[:220])
        if len(snippets) >= limit:
            break
    return snippets


def _quantified_ratio(text: str) -> float:
    claims = _extract_candidate_claims(text, limit=80)
    if not claims:
        return 0.0
    quantified = sum(1 for claim in claims if _METRIC_RE.search(claim))
    return quantified / max(1, len(claims))


def _repetition_ratio(text: str) -> float:
    claims = _extract_candidate_claims(text, limit=80)
    if len(claims) < 2:
        return 0.0
    duplicates = 0
    for idx, a in enumerate(claims):
        for b in claims[idx + 1 :]:
            if SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.88:
                duplicates += 1
    return duplicates / max(1, len(claims))


def _risk_from_score(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"


def _llm_json_with_policy(*, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    strict = _is_strict_mode()
    if strict:
        if not tools_llm_enabled():
            raise RecruiterQualityError("Recruiter quality mode is enabled but OpenAI is not configured.", status_code=503)
        try:
            return json_completion_required(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2, max_output_tokens=1300)
        except ToolsLLMError as exc:
            raise RecruiterQualityError(str(exc), status_code=503) from exc
    return json_completion(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2, max_output_tokens=1300)


def _parse_role_level(jd_text: str) -> tuple[RoleLevelInference, str]:
    lower = jd_text.lower()
    hits: list[tuple[RoleLevelInference, int]] = []
    for level, terms in _ROLE_KEYWORDS.items():
        if level == "Unclear":
            continue
        count = sum(1 for term in terms if term in lower)
        if count > 0:
            hits.append((level, count))
    if not hits:
        return "Unclear", "No explicit seniority markers were found in the JD."
    hits.sort(key=lambda item: item[1], reverse=True)
    top_level = hits[0][0]
    reason = f"Detected seniority markers aligned with {top_level} level requirements."
    return top_level, reason


def _serialize_examples(items: list[str]) -> list[str]:
    return [item.strip()[:220] for item in items if item and item.strip()][:3]


def _dimension_level(score: float) -> str:
    if score >= 0.67:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"


def _overall_signal_level(levels: list[str]) -> str:
    if not levels:
        return "Weak"
    score_map = {"Low": 1, "Medium": 2, "High": 3}
    avg = sum(score_map.get(level, 1) for level in levels) / len(levels)
    if avg >= 2.5:
        return "Strong"
    if avg >= 1.75:
        return "Moderate"
    return "Weak"


def _extract_jd_requirement_lists(jd_text: str) -> tuple[list[str], list[str]]:
    must_haves: list[str] = []
    nice_to_haves: list[str] = []
    for raw_line in _split_lines(jd_text):
        line = raw_line.strip(" -\t")
        lower = line.lower()
        if len(line) < 3:
            continue
        if any(marker in lower for marker in _MUST_HAVE_MARKERS):
            must_haves.append(line[:120])
        elif any(marker in lower for marker in _NICE_TO_HAVE_MARKERS):
            nice_to_haves.append(line[:120])

    if not must_haves:
        must_haves = _top_terms(jd_text, limit=10)
    if not nice_to_haves:
        tail_terms = _top_terms(jd_text, limit=25)
        nice_to_haves = [term for term in tail_terms if term not in must_haves][:10]

    return must_haves[:15], nice_to_haves[:15]


def _authenticity_fallback(payload: ResumeAuthenticityRequest) -> ResumeAuthenticityResponse:
    resume = payload.resume_text
    jd = payload.jd_text or ""
    lower = resume.lower()
    cliche_hits = [phrase for phrase in _CICHE_PHRASES if phrase in lower]
    quant_ratio = _quantified_ratio(resume)
    repeat_ratio = _repetition_ratio(resume)

    jd_terms = _top_terms(jd, limit=15) if jd else []
    resume_terms = set(_top_terms(resume, limit=200))
    missing_terms = [term for term in jd_terms if term not in resume_terms][:5]

    score = 0.0
    score += min(0.45, len(cliche_hits) * 0.08)
    score += min(0.3, repeat_ratio * 2.2)
    score += min(0.4, max(0.0, 0.5 - quant_ratio))
    score += min(0.25, len(missing_terms) * 0.04)
    risk_level = _risk_from_score(min(1.0, score))

    signals: list[ResumeAuthenticitySignal] = []
    if cliche_hits:
        signals.append(
            ResumeAuthenticitySignal(
                name="AI-like phrasing",
                severity="Medium" if len(cliche_hits) < 4 else "High",
                explanation="Repeated generic phrasing lowers recruiter trust in ownership and specificity.",
                examples=_serialize_examples(_find_snippets(resume, cliche_hits, limit=3)),
                suggested_fix="Replace generic phrases with concrete outcomes and scope from your own projects.",
            )
        )
    if quant_ratio < 0.35:
        signals.append(
            ResumeAuthenticitySignal(
                name="Vague achievements",
                severity="High" if quant_ratio < 0.2 else "Medium",
                explanation="Many achievement lines lack measurable outcomes, making claims harder to verify.",
                examples=_serialize_examples(_extract_candidate_claims(resume, limit=3)),
                suggested_fix="Add one metric per major claim: scale, speed, cost, reliability, or business impact.",
            )
        )
    if repeat_ratio > 0.15:
        signals.append(
            ResumeAuthenticitySignal(
                name="Repetition risk",
                severity="Medium",
                explanation="Near-duplicate bullets can look template-driven and reduce perceived depth.",
                examples=_serialize_examples(_extract_candidate_claims(resume, limit=3)),
                suggested_fix="Merge repeated bullets and keep one unique outcome per line.",
            )
        )
    if missing_terms:
        signals.append(
            ResumeAuthenticitySignal(
                name="Keyword stuffing risk",
                severity="Low" if len(missing_terms) <= 2 else "Medium",
                explanation="Large role-term gaps can trigger overstuffing attempts when tailoring.",
                examples=_serialize_examples(missing_terms),
                suggested_fix="Only add missing terms where you can provide direct evidence in experience bullets.",
            )
        )
    if not signals:
        signals.append(
            ResumeAuthenticitySignal(
                name="Evidence consistency",
                severity="Low",
                explanation="The resume shows consistent language and measurable outcomes across key claims.",
                examples=_serialize_examples(_extract_candidate_claims(resume, limit=2)),
                suggested_fix="Keep the same evidence-first writing style while tailoring for each role.",
            )
        )

    summary = (
        "This assessment found a mixed authenticity signal profile. "
        "The resume includes strong technical context, but some claims can be tightened with clearer evidence. "
        "Treat the risk score as a verification cue, not a final judgment."
    )
    if risk_level == "Low":
        summary = (
            "This resume presents a generally credible profile with mostly specific claims. "
            "Minor improvements can still increase verification confidence during recruiter review."
        )
    elif risk_level == "High":
        summary = (
            "This resume shows several high-risk authenticity indicators, mainly around specificity and repeat patterns. "
            "Prioritize evidence-backed edits before using it for screening-heavy roles."
        )

    return ResumeAuthenticityResponse(
        risk_level=risk_level,
        overall_summary=summary,
        signals=signals[:6],
        disclaimers=[
            "This is probabilistic; do not treat as proof.",
            "Use this as decision support and validate with human review.",
        ],
        generated_at=_utc_now(),
    )


def run_resume_authenticity(payload: ResumeAuthenticityRequest) -> ResumeAuthenticityResponse:
    fallback = _authenticity_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are assisting a recruiter. Be concise, factual, and evidence-grounded. "
            "Never claim certainty about AI-written resumes. Return strict JSON only."
        ),
        user_prompt=(
            "Analyze resume authenticity risk with probabilistic language.\n"
            f"Resume:\n{payload.resume_text[:10000]}\n\n"
            f"JD (optional):\n{payload.jd_text[:7000] if payload.jd_text else '(not provided)'}\n\n"
            "Return JSON keys exactly: risk_level, overall_summary, signals, disclaimers.\n"
            "signals must be array of objects with keys: name, severity, explanation, examples, suggested_fix.\n"
            "severity and risk_level must be Low/Medium/High."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        response = ResumeAuthenticityResponse(
            risk_level=llm_payload.get("risk_level", fallback.risk_level),
            overall_summary=str(llm_payload.get("overall_summary", fallback.overall_summary)),
            signals=llm_payload.get("signals", [signal.model_dump() for signal in fallback.signals]),
            disclaimers=llm_payload.get("disclaimers", fallback.disclaimers),
            generated_at=_utc_now(),
        )
        if "probabilistic" not in " ".join(response.disclaimers).lower():
            response.disclaimers.append("This is probabilistic; do not treat as proof.")
        return response
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Authenticity analysis did not meet required JSON quality.", status_code=422) from exc
        return fallback


def _claim_verification_fallback(payload: ClaimVerificationRequest) -> ClaimVerificationResponse:
    claims = _extract_candidate_claims(payload.resume_text, limit=7)
    rows = []
    red_flags: list[str] = []
    for claim in claims:
        has_metric = bool(_METRIC_RE.search(claim))
        why = "This claim influences role fit and should be validated for depth, ownership, and context."
        questions = [
            f"What was the specific business or technical context behind: '{claim[:90]}'?",
            "What trade-offs did you evaluate, and why did you pick your final approach?",
            "How did you measure success and what was the baseline before your change?",
        ]
        if has_metric:
            questions.append("How was this metric measured and how stable was the improvement over time?")
        else:
            questions.append("Can you provide a concrete metric that demonstrates the impact?")
            red_flags.append(f"Follow up on unquantified claim: {claim[:110]}")
        questions.append("What failed during implementation, and what did you change afterward?")
        rows.append(
            {
                "claim_text": claim,
                "why_verify": why,
                "questions": questions[:5],
                "strong_answer_signals": [
                    "Provides measurable baseline and outcome.",
                    "Explains constraints and technical trade-offs clearly.",
                    "Describes own decisions and lessons learned.",
                ],
                "weak_answer_signals": [
                    "Uses generic language without specific context.",
                    "Cannot explain measurement method or baseline.",
                    "Cannot describe failures or trade-off decisions.",
                ],
            }
        )
    if not rows:
        rows.append(
            {
                "claim_text": "General experience summary",
                "why_verify": "The resume has limited claim granularity; ask deeper follow-up questions.",
                "questions": [
                    "What was your highest-impact project in the last 12 months?",
                    "What measurable outcome did you personally influence most?",
                    "What technical decision did you own end-to-end?",
                ],
                "strong_answer_signals": [
                    "Specific ownership and scope.",
                    "Clear measurable outcome.",
                    "Concrete technical trade-offs.",
                ],
                "weak_answer_signals": [
                    "Vague role description.",
                    "No measurable outcomes.",
                    "No decision ownership.",
                ],
            }
        )
    return ClaimVerificationResponse(
        summary="Use these targeted questions to verify real depth behind each claim and reduce screening uncertainty.",
        claims=rows,
        red_flag_follow_ups=red_flags[:8],
        generated_at=_utc_now(),
    )


def run_claim_verification(payload: ClaimVerificationRequest) -> ClaimVerificationResponse:
    fallback = _claim_verification_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are assisting a recruiter in structured claim verification. "
            "Generate concrete, evidence-linked interview questions. Return strict JSON only."
        ),
        user_prompt=(
            f"Resume:\n{payload.resume_text[:12000]}\n\n"
            f"JD:\n{payload.jd_text[:7000] if payload.jd_text else '(not provided)'}\n\n"
            "Return JSON keys: summary, claims, red_flag_follow_ups.\n"
            "claims items must include claim_text, why_verify, questions(3-5), strong_answer_signals(3), weak_answer_signals(3)."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return ClaimVerificationResponse(
            summary=str(llm_payload.get("summary", fallback.summary)),
            claims=llm_payload.get("claims", [claim.model_dump() for claim in fallback.claims]),
            red_flag_follow_ups=llm_payload.get("red_flag_follow_ups", fallback.red_flag_follow_ups),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Claim verification output did not pass strict quality checks.", status_code=422) from exc
        return fallback


def _jd_quality_fallback(payload: JDQualityAnalyzerRequest) -> JDQualityResponse:
    jd = payload.jd_text
    lines = _split_lines(jd)
    lower = jd.lower()
    issues: list[JDQualityIssue] = []

    level, level_reason = _parse_role_level(jd)
    if ("junior" in lower and "senior" in lower) or ("entry" in lower and "lead" in lower):
        issues.append(
            JDQualityIssue(
                category="Mixed seniority",
                severity="High",
                evidence="JD contains conflicting seniority markers (e.g., junior and senior).",
                why_it_hurts="Conflicting expectations reduce qualified applications and increase noisy pipelines.",
                improvement_suggestion="Separate role levels or clarify a single target seniority range.",
            )
        )
    long_lines = [line for line in lines if len(line) > 180]
    if long_lines:
        issues.append(
            JDQualityIssue(
                category="Laundry list",
                severity="Medium",
                evidence=long_lines[0][:180],
                why_it_hurts="Dense requirement lists reduce readability and candidate confidence.",
                improvement_suggestion="Group requirements into must-have and nice-to-have with concise bullets.",
            )
        )
    vague_lines = [line for line in lines if any(term in line.lower() for term in ("various", "etc", "other duties", "assist with"))]
    if vague_lines:
        issues.append(
            JDQualityIssue(
                category="Vague responsibilities",
                severity="Medium",
                evidence=vague_lines[0][:180],
                why_it_hurts="Vague responsibilities hide role scope and make candidate self-selection harder.",
                improvement_suggestion="Rewrite vague lines into concrete responsibilities with expected outcomes.",
            )
        )
    if len(_top_terms(jd, limit=60)) > 45:
        issues.append(
            JDQualityIssue(
                category="Unrealistic scope",
                severity="High",
                evidence="JD contains an unusually broad skill spread for a single role.",
                why_it_hurts="Over-scoped roles discourage strong candidates and lower relevance.",
                improvement_suggestion="Prioritize top 6-10 critical capabilities and move the rest to optional.",
            )
        )

    if not issues:
        issues.append(
            JDQualityIssue(
                category="Clarity baseline",
                severity="Low",
                evidence="The JD structure is mostly clear with consistent role signals.",
                why_it_hurts="Low friction descriptions still benefit from concise must-have/optional separation.",
                improvement_suggestion="Add a short role-success section (first 90 days) to improve signal quality.",
            )
        )

    rating = "Clear"
    if any(issue.severity == "High" for issue in issues):
        rating = "Problematic"
    elif any(issue.severity == "Medium" for issue in issues):
        rating = "Risky"

    summary = "The JD is generally usable but has quality gaps that could reduce candidate quality and screening precision."
    if rating == "Clear":
        summary = "The JD is mostly clear and actionable. Minor refinements can improve conversion and interview quality."
    elif rating == "Problematic":
        summary = "The JD has structural quality risks that are likely to attract mismatched applicants."

    return JDQualityResponse(
        rating=rating,
        summary=summary,
        issues=issues[:8],
        role_level_inference=level,
        role_level_reasoning=level_reason,
        generated_at=_utc_now(),
    )


def run_jd_quality(payload: JDQualityAnalyzerRequest) -> JDQualityResponse:
    fallback = _jd_quality_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are a recruiter operations assistant. Evaluate JD quality with concise evidence and incremental improvements. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"JD:\n{payload.jd_text[:14000]}\n\n"
            "Return JSON keys: rating, summary, issues, role_level_inference, role_level_reasoning.\n"
            "issues items require: category, severity, evidence, why_it_hurts, improvement_suggestion."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return JDQualityResponse(
            rating=llm_payload.get("rating", fallback.rating),
            summary=str(llm_payload.get("summary", fallback.summary)),
            issues=llm_payload.get("issues", [issue.model_dump() for issue in fallback.issues]),
            role_level_inference=llm_payload.get("role_level_inference", fallback.role_level_inference),
            role_level_reasoning=str(llm_payload.get("role_level_reasoning", fallback.role_level_reasoning)),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("JD quality analyzer returned invalid strict JSON.", status_code=422) from exc
        return fallback


def _ats_human_fallback(payload: ATSHumanRiskSplitRequest) -> ATSHumanRiskSplitResponse:
    resume = payload.resume_text
    jd = payload.jd_text or ""
    quant_ratio = _quantified_ratio(resume)
    repeat_ratio = _repetition_ratio(resume)
    jd_terms = _top_terms(jd, limit=20) if jd else []
    resume_terms = set(_top_terms(resume, limit=200))
    hard_filter_gaps = [term for term in jd_terms if term not in resume_terms][:5]

    ats_score = min(1.0, max(0.0, len(hard_filter_gaps) * 0.12 + max(0.0, 0.35 - quant_ratio)))
    human_score = min(1.0, max(0.0, len([p for p in _CICHE_PHRASES if p in resume.lower()]) * 0.1 + repeat_ratio * 2))

    ats_level = _risk_from_score(ats_score)
    human_level = _risk_from_score(human_score)

    ats_drivers = []
    if hard_filter_gaps:
        ats_drivers.append(f"Potential hard-filter gaps: {', '.join(hard_filter_gaps[:3])}.")
    ats_drivers.append("Keyword coverage and section clarity affect searchability in ATS systems.")
    ats_drivers.append("File structure and concise section naming improve parse reliability.")
    ats_drivers = ats_drivers[:5]

    human_drivers = []
    if quant_ratio < 0.35:
        human_drivers.append("Low measurable impact density can reduce perceived credibility.")
    if repeat_ratio > 0.12:
        human_drivers.append("Repetition across bullets may suggest shallow project depth.")
    human_drivers.append("Recruiters prioritize specific ownership, scope, and outcomes.")
    human_drivers = human_drivers[:5]

    quick_wins = [
        QuickWin(item="Add one measurable outcome to top three experience bullets.", effort_minutes=25, impact="High"),
        QuickWin(item="Align summary headline with target role and top domain terms.", effort_minutes=12, impact="Med"),
        QuickWin(item="Consolidate repetitive bullets into unique impact statements.", effort_minutes=18, impact="Med"),
    ]

    focus = "Both"
    if ats_level == "High" and human_level != "High":
        focus = "Fix ATS first"
    elif human_level == "High" and ats_level != "High":
        focus = "Fix credibility first"

    return ATSHumanRiskSplitResponse(
        ats_risk=RiskBlock(
            level=ats_level,
            explanation="ATS risk is based on likely parsing/findability friction and role-term coverage.",
            top_drivers=ats_drivers[:5] or ["No major ATS blockers detected."],
        ),
        human_risk=RiskBlock(
            level=human_level,
            explanation="Human risk reflects recruiter trust signals: specificity, evidence depth, and clarity.",
            top_drivers=human_drivers[:5] or ["No major credibility blockers detected."],
        ),
        quick_wins=quick_wins,
        where_to_focus=focus,
        hard_filter_gaps=hard_filter_gaps,
        generated_at=_utc_now(),
    )


def run_ats_vs_human(payload: ATSHumanRiskSplitRequest) -> ATSHumanRiskSplitResponse:
    fallback = _ats_human_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You assist recruiters in separating ATS risk from human review risk. "
            "Be evidence-grounded and avoid deterministic ATS vendor claims. Return strict JSON only."
        ),
        user_prompt=(
            f"Resume:\n{payload.resume_text[:12000]}\n\n"
            f"JD:\n{payload.jd_text[:7000] if payload.jd_text else '(not provided)'}\n\n"
            "Return JSON keys: ats_risk, human_risk, quick_wins, where_to_focus, hard_filter_gaps.\n"
            "ats_risk/human_risk must include: level, explanation, top_drivers.\n"
            "quick_wins items: item, effort_minutes, impact(Low/Med/High)."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return ATSHumanRiskSplitResponse(
            ats_risk=llm_payload.get("ats_risk", fallback.ats_risk.model_dump()),
            human_risk=llm_payload.get("human_risk", fallback.human_risk.model_dump()),
            quick_wins=llm_payload.get("quick_wins", [item.model_dump() for item in fallback.quick_wins]),
            where_to_focus=str(llm_payload.get("where_to_focus", fallback.where_to_focus)),
            hard_filter_gaps=llm_payload.get("hard_filter_gaps", fallback.hard_filter_gaps),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("ATS vs Human output did not satisfy strict quality checks.", status_code=422) from exc
        return fallback


def _fit_level(score: int) -> str:
    if score >= 70:
        return "Strong"
    if score >= 45:
        return "Moderate"
    return "Weak"


def _resume_compare_fallback(payload: ResumeCompareRequest) -> ResumeCompareResponse:
    jd_terms = set(_top_terms(payload.jd_text, limit=35))
    candidates: list[ResumeCompareCandidateOutput] = []
    scored: list[tuple[str, int]] = []

    for item in payload.resumes:
        resume_terms = set(_top_terms(item.resume_text, limit=220))
        overlap = len(jd_terms.intersection(resume_terms))
        denominator = max(1, len(jd_terms))
        score = int((overlap / denominator) * 100)
        quant_ratio = _quantified_ratio(item.resume_text)
        repeat_ratio = _repetition_ratio(item.resume_text)
        ats_level = _risk_from_score(max(0.0, 0.7 - (score / 100)))
        human_level = _risk_from_score(max(0.0, (0.45 - quant_ratio) + repeat_ratio))

        strengths = [
            f"JD term overlap: {overlap}/{denominator}.",
            "Resume structure includes clear project and experience progression.",
            "Technical language mostly aligned to target role scope.",
        ]
        if quant_ratio >= 0.35:
            strengths.append("Includes measurable impact in key experience bullets.")
        risks = [
            "Some JD priorities are not directly evidenced in recent experience.",
            "Interview depth checks needed for highest-impact claims.",
        ]
        if quant_ratio < 0.3:
            risks.append("Limited quantifiable outcomes in experience bullets.")
        interview_focus = [
            "Ownership scope and decision trade-offs.",
            "Measurement method behind key project outcomes.",
            "Failure handling and post-release iteration.",
            "Role-specific depth in required stack terms.",
        ]
        candidates.append(
            ResumeCompareCandidateOutput(
                label=item.candidate_label,
                fit_level=_fit_level(score),
                ats_risk=ats_level,
                human_risk=human_level,
                strengths=strengths[:5],
                risks=risks[:5],
                interview_focus=interview_focus[:5],
            )
        )
        scored.append((item.candidate_label, score))

    scored.sort(key=lambda row: row[1], reverse=True)
    shortlist = [label for label, _ in scored[:2]]
    summary = (
        "Candidate fit differs by role-term alignment, measurable impact clarity, and claim depth. "
        "Use the shortlist as interview prioritization guidance, not final hiring automation."
    )
    return ResumeCompareResponse(
        comparison_summary=summary,
        candidates=candidates,
        recommendation=f"Shortlist {', '.join(shortlist)} for the next round with focused verification questions.",
        decision_log=(
            f"Initial shortlist based on JD alignment and risk split: {', '.join(shortlist)}. "
            "Final decision requires structured interviews and evidence checks."
        ),
        generated_at=_utc_now(),
    )


def run_resume_compare(payload: ResumeCompareRequest) -> ResumeCompareResponse:
    fallback = _resume_compare_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are a recruiter evaluation assistant. Compare candidate resumes against one JD. "
            "Be explainable and avoid absolute winner language. Return strict JSON only."
        ),
        user_prompt=(
            f"JD:\n{payload.jd_text[:9000]}\n\n"
            f"Candidates:\n{[{'label': item.candidate_label, 'resume_text': item.resume_text[:6000]} for item in payload.resumes]}\n\n"
            "Return JSON keys: comparison_summary, candidates, recommendation, decision_log.\n"
            "Each candidate item: label, fit_level(Strong/Moderate/Weak), ats_risk, human_risk, strengths, risks, interview_focus."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return ResumeCompareResponse(
            comparison_summary=str(llm_payload.get("comparison_summary", fallback.comparison_summary)),
            candidates=llm_payload.get("candidates", [candidate.model_dump() for candidate in fallback.candidates]),
            recommendation=str(llm_payload.get("recommendation", fallback.recommendation)),
            decision_log=str(llm_payload.get("decision_log", fallback.decision_log)),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Resume comparison output failed strict quality validation.", status_code=422) from exc
        return fallback


def _resume_signal_strength_fallback(payload: ResumeSignalStrengthRequest) -> ResumeSignalStrengthResponse:
    resume = payload.resume_text
    jd = payload.jd_text or ""
    claims = _extract_candidate_claims(resume, limit=100)
    lines = _split_lines(resume)
    lower = resume.lower()

    quant_ratio = _quantified_ratio(resume)
    repeat_ratio = _repetition_ratio(resume)

    ownership_terms = ("owned", "led", "drove", "spearheaded", "managed", "architected")
    ownership_hits = sum(1 for line in lines if any(term in line.lower() for term in ownership_terms))
    ownership_ratio = ownership_hits / max(1, len(claims))

    scope_terms = ("users", "customers", "regions", "countries", "teams", "services", "tenants", "platform")
    scope_hits = sum(1 for line in lines if any(term in line.lower() for term in scope_terms))
    scope_ratio = scope_hits / max(1, len(claims))

    technical_terms = (
        "python",
        "java",
        "node",
        "react",
        "aws",
        "azure",
        "kubernetes",
        "docker",
        "microservice",
        "architecture",
        "postgres",
        "mongodb",
        "graphql",
    )
    technical_hits = len([term for term in technical_terms if term in lower])
    technical_ratio = min(1.0, technical_hits / 8.0)

    communication_terms = ("stakeholder", "collaborat", "present", "communicat", "cross-functional", "partner")
    communication_hits = sum(1 for line in lines if any(term in line.lower() for term in communication_terms))
    communication_ratio = communication_hits / max(1, len(claims))

    consistency_score = max(0.0, 1.0 - min(0.9, repeat_ratio * 2.5))

    dimensions = [
        ResumeSignalDimension(
            name="Impact",
            level=_dimension_level(quant_ratio),
            evidence=_serialize_examples(_find_snippets(resume, ["%", "reduced", "improved", "increased", "saved"], limit=3)),
            why_it_matters="Impact evidence helps recruiters compare outcomes instead of just responsibilities.",
            improvement_hint="Add measurable before/after outcomes for high-priority projects.",
        ),
        ResumeSignalDimension(
            name="Ownership",
            level=_dimension_level(ownership_ratio),
            evidence=_serialize_examples(_find_snippets(resume, list(ownership_terms), limit=3)),
            why_it_matters="Ownership signals show decision authority and independent execution.",
            improvement_hint="Clarify what decisions you owned versus team-level contributions.",
        ),
        ResumeSignalDimension(
            name="Scope",
            level=_dimension_level(scope_ratio),
            evidence=_serialize_examples(_find_snippets(resume, list(scope_terms), limit=3)),
            why_it_matters="Scope helps estimate complexity and the scale of delivered work.",
            improvement_hint="Mention scale indicators such as users, systems, revenue, or team size.",
        ),
        ResumeSignalDimension(
            name="Technical Depth",
            level=_dimension_level(technical_ratio),
            evidence=_serialize_examples(_find_snippets(resume, list(technical_terms), limit=3)),
            why_it_matters="Depth indicates whether the candidate can handle role-critical complexity.",
            improvement_hint="Add stack-specific decisions, constraints, and trade-offs for key projects.",
        ),
        ResumeSignalDimension(
            name="Communication",
            level=_dimension_level(communication_ratio),
            evidence=_serialize_examples(_find_snippets(resume, list(communication_terms), limit=3)),
            why_it_matters="Communication quality is a strong predictor of cross-functional effectiveness.",
            improvement_hint="Include examples of stakeholder alignment, handoffs, or documentation impact.",
        ),
        ResumeSignalDimension(
            name="Consistency",
            level=_dimension_level(consistency_score),
            evidence=_serialize_examples(_extract_candidate_claims(resume, limit=3)),
            why_it_matters="Consistent structure and claim quality improve screening trust.",
            improvement_hint="Keep one unique outcome per bullet and avoid near-duplicate statements.",
        ),
    ]

    role_fit_notes: list[str] = []
    if jd.strip():
        jd_terms = _top_terms(jd, limit=15)
        resume_terms = set(_top_terms(resume, limit=200))
        matched = [term for term in jd_terms if term in resume_terms]
        missing = [term for term in jd_terms if term not in resume_terms]
        role_fit_notes.append(
            f"Matched role terms: {', '.join(matched[:5]) if matched else 'limited direct overlap'}."
        )
        if missing:
            role_fit_notes.append(f"Potentially under-evidenced terms: {', '.join(missing[:5])}.")
        if quant_ratio < 0.35:
            role_fit_notes.append("Role alignment may be weakened by low measurable-impact density.")
        if repeat_ratio > 0.15:
            role_fit_notes.append("Repetition can reduce perceived depth for role-critical experience.")

    overall = _overall_signal_level([dimension.level for dimension in dimensions])
    summary = (
        "Signal strength is moderate overall, with clearer strengths in technical context than in outcome evidence. "
        "Use the dimension-level notes to tighten verification readiness before interviews."
    )
    if overall == "Strong":
        summary = (
            "Signal strength is strong across most dimensions, with clear evidence of impact, scope, and ownership. "
            "Only minor refinements are needed for interview readiness."
        )
    elif overall == "Weak":
        summary = (
            "Signal strength is weak in key dimensions, mainly due to limited measurable outcomes and ownership detail. "
            "Prioritize evidence-first edits before screening."
        )

    return ResumeSignalStrengthResponse(
        overall_signal_level=overall,
        summary=summary,
        signal_dimensions=dimensions,
        role_fit_notes=role_fit_notes[:5],
        generated_at=_utc_now(),
    )


def run_resume_signal_strength(payload: ResumeSignalStrengthRequest) -> ResumeSignalStrengthResponse:
    fallback = _resume_signal_strength_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are assisting a recruiter. Produce evidence-grounded resume signal analysis without hype. "
            "Never invent facts. Return strict JSON only."
        ),
        user_prompt=(
            f"Resume:\n{payload.resume_text[:12000]}\n\n"
            f"JD (optional):\n{payload.jd_text[:7000] if payload.jd_text else '(not provided)'}\n\n"
            "Return JSON keys: overall_signal_level, summary, signal_dimensions, role_fit_notes.\n"
            "signal_dimensions must contain exactly names: Impact, Ownership, Scope, Technical Depth, Communication, Consistency.\n"
            "Each dimension: name, level(Low/Medium/High), evidence(max3), why_it_matters, improvement_hint.\n"
            "overall_signal_level: Strong/Moderate/Weak."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return ResumeSignalStrengthResponse(
            overall_signal_level=llm_payload.get("overall_signal_level", fallback.overall_signal_level),
            summary=str(llm_payload.get("summary", fallback.summary)),
            signal_dimensions=llm_payload.get(
                "signal_dimensions",
                [item.model_dump() for item in fallback.signal_dimensions],
            ),
            role_fit_notes=llm_payload.get("role_fit_notes", fallback.role_fit_notes),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Resume signal strength output failed strict quality validation.", status_code=422) from exc
        return fallback


def _jd_market_reality_fallback(payload: JDMarketRealityRequest) -> JDMarketRealityResponse:
    jd = payload.jd_text
    lines = _split_lines(jd)
    lower = jd.lower()
    inferred_level, level_reason = _parse_role_level(jd)

    must_haves, nice_to_haves = _extract_jd_requirement_lists(jd)
    concerns: list[JDMarketConcern] = []

    if len(must_haves) >= 12:
        concerns.append(
            JDMarketConcern(
                concern="too many must-haves",
                severity="High",
                evidence=must_haves[0][:180],
                impact="Large must-have sets often reduce qualified applicants and slow hiring.",
                suggestion="Move lower-priority requirements into nice-to-have to widen relevant pipelines.",
            )
        )
    if ("junior" in lower and "senior" in lower) or ("entry" in lower and "lead" in lower):
        concerns.append(
            JDMarketConcern(
                concern="conflicting seniority signals",
                severity="High",
                evidence="JD contains mixed level markers such as junior and senior/lead.",
                impact="Conflicting level language causes candidate confusion and lower response quality.",
                suggestion="Set one target level and align responsibilities and expectations to it.",
            )
        )
    if len(_top_terms(jd, limit=80)) >= 55:
        concerns.append(
            JDMarketConcern(
                concern="scope too broad",
                severity="Medium",
                evidence=lines[0][:180] if lines else "Broad skill spread detected across the JD.",
                impact="Over-broad roles can increase screening load and reduce fit precision.",
                suggestion="Prioritize core outcomes and keep secondary skills optional.",
            )
        )
    vague_markers = ("various", "etc.", "other duties", "as needed", "dynamic tasks")
    vague_line = next((line for line in lines if any(marker in line.lower() for marker in vague_markers)), "")
    if vague_line:
        concerns.append(
            JDMarketConcern(
                concern="vague responsibilities",
                severity="Medium",
                evidence=vague_line[:180],
                impact="Vague expectations make self-selection harder for strong applicants.",
                suggestion="Convert vague lines into concrete deliverables with expected outcomes.",
            )
        )

    if not concerns:
        concerns.append(
            JDMarketConcern(
                concern="market realism baseline",
                severity="Low",
                evidence=(lines[0][:180] if lines else "JD wording is generally clear."),
                impact="The role description is mostly realistic and operationally clear.",
                suggestion="Minor improvements: clarify interview criteria and first-90-day outcomes.",
            )
        )

    realism = "Reasonable"
    if any(item.severity == "High" for item in concerns):
        realism = "Unrealistic"
    elif any(item.severity == "Medium" for item in concerns):
        realism = "Stretch"

    return JDMarketRealityResponse(
        realism_rating=realism,
        inferred_role_level=inferred_level,
        role_level_reasoning=level_reason,
        concerns=concerns[:8],
        must_have_vs_nice_to_have=MustHaveNiceToHaveBreakdown(
            must_have_candidates=must_haves[:15],
            nice_to_have_candidates=nice_to_haves[:15],
        ),
        generated_at=_utc_now(),
    )


def run_jd_market_reality(payload: JDMarketRealityRequest) -> JDMarketRealityResponse:
    fallback = _jd_market_reality_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are a recruiter operations assistant. Evaluate JD realism with practical, evidence-grounded guidance. "
            "Do not invent market statistics. Return strict JSON only."
        ),
        user_prompt=(
            f"JD:\n{payload.jd_text[:14000]}\n\n"
            "Return JSON keys: realism_rating, inferred_role_level, role_level_reasoning, concerns, must_have_vs_nice_to_have.\n"
            "concerns items: concern, severity, evidence, impact, suggestion.\n"
            "must_have_vs_nice_to_have has keys must_have_candidates and nice_to_have_candidates."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return JDMarketRealityResponse(
            realism_rating=llm_payload.get("realism_rating", fallback.realism_rating),
            inferred_role_level=llm_payload.get("inferred_role_level", fallback.inferred_role_level),
            role_level_reasoning=str(llm_payload.get("role_level_reasoning", fallback.role_level_reasoning)),
            concerns=llm_payload.get("concerns", [item.model_dump() for item in fallback.concerns]),
            must_have_vs_nice_to_have=llm_payload.get(
                "must_have_vs_nice_to_have",
                fallback.must_have_vs_nice_to_have.model_dump(),
            ),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("JD market reality output failed strict quality checks.", status_code=422) from exc
        return fallback


def _role_seniority_definition_fallback(payload: RoleSeniorityDefinitionRequest) -> RoleSeniorityDefinitionResponse:
    jd = payload.jd_text
    lines = _split_lines(jd)
    inferred_level, level_reason = _parse_role_level(jd)

    def collect(markers: tuple[str, ...]) -> list[str]:
        hits = [line[:140] for line in lines if any(marker in line.lower() for marker in markers)]
        return hits[:6]

    leadership = collect(_LEADERSHIP_MARKERS)
    architecture = collect(_ARCH_MARKERS)
    autonomy = collect(_AUTONOMY_MARKERS)
    execution = collect(_EXECUTION_MARKERS)

    marker_total = len(leadership) + len(architecture) + len(autonomy) + len(execution)
    confidence = "Low"
    if inferred_level != "Unclear" and marker_total >= 6:
        confidence = "High"
    elif inferred_level != "Unclear" or marker_total >= 3:
        confidence = "Medium"

    rationale = [
        f"Inferred level: {inferred_level}. {level_reason}",
        f"Leadership signals detected: {len(leadership)}.",
        f"Architecture/system-design signals detected: {len(architecture)}.",
        f"Autonomy/ownership signals detected: {len(autonomy)}.",
        f"Execution and delivery signals detected: {len(execution)}.",
    ]
    evidence_line = next((line for line in lines if line), "")
    if evidence_line:
        rationale.append(f"Representative JD evidence: {evidence_line[:130]}")

    focus_by_level = {
        "Junior": ["Fundamentals and learning velocity", "Code quality basics", "Task ownership with guidance"],
        "Mid": ["Independent feature delivery", "Debugging and reliability", "Cross-team collaboration"],
        "Senior": ["System design trade-offs", "Mentoring and code review quality", "Production incident leadership"],
        "Lead": ["Team direction and delivery planning", "Architecture decisions", "Stakeholder alignment"],
        "Staff": ["Org-level technical strategy", "Cross-domain architecture", "Influence without authority"],
        "Unclear": ["Clarify target level first", "Set role outcomes", "Define interview depth per competency"],
    }

    return RoleSeniorityDefinitionResponse(
        recommended_level=inferred_level,
        confidence=confidence,
        rationale=rationale[:8],
        signals_detected=RoleSenioritySignals(
            leadership_signals=leadership,
            architecture_signals=architecture,
            autonomy_signals=autonomy,
            execution_signals=execution,
        ),
        suggested_interview_focus=focus_by_level[inferred_level][:10],
        generated_at=_utc_now(),
    )


def run_role_seniority_definition(payload: RoleSeniorityDefinitionRequest) -> RoleSeniorityDefinitionResponse:
    fallback = _role_seniority_definition_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You are assisting a recruiter in defining role seniority from a JD. "
            "Be evidence-grounded and explain uncertainty. Return strict JSON only."
        ),
        user_prompt=(
            f"JD:\n{payload.jd_text[:14000]}\n\n"
            "Return JSON keys: recommended_level, confidence, rationale, signals_detected, suggested_interview_focus.\n"
            "signals_detected keys: leadership_signals, architecture_signals, autonomy_signals, execution_signals.\n"
            "confidence values: Low/Medium/High. recommended_level: Junior/Mid/Senior/Lead/Staff/Unclear."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return RoleSeniorityDefinitionResponse(
            recommended_level=llm_payload.get("recommended_level", fallback.recommended_level),
            confidence=llm_payload.get("confidence", fallback.confidence),
            rationale=llm_payload.get("rationale", fallback.rationale),
            signals_detected=llm_payload.get("signals_detected", fallback.signals_detected.model_dump()),
            suggested_interview_focus=llm_payload.get("suggested_interview_focus", fallback.suggested_interview_focus),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Role seniority definition output failed strict validation.", status_code=422) from exc
        return fallback


def _shortlist_justification_fallback(payload: ShortlistJustificationRequest) -> ShortlistJustificationResponse:
    jd_terms = set(_top_terms(payload.jd_text, limit=35))
    notes: list[ShortlistCandidateNote] = []
    scored: list[tuple[str, int]] = []

    for candidate in payload.candidates:
        resume = candidate.resume_text
        terms = set(_top_terms(resume, limit=220))
        overlap = len(jd_terms.intersection(terms))
        base = max(1, len(jd_terms))
        score = int((overlap / base) * 100)
        quant_ratio = _quantified_ratio(resume)
        repeat_ratio = _repetition_ratio(resume)

        fit_level = _fit_level(score)
        strengths = [
            f"JD overlap terms: {overlap}/{base}.",
            "Resume shows relevant domain vocabulary for target role.",
            "Experience progression supports interview viability.",
        ]
        if quant_ratio > 0.35:
            strengths.append("Includes measurable outcomes in experience bullets.")

        risks = [
            "Some required capabilities are not strongly evidenced in recent experience.",
            "Interview depth checks are needed for high-impact claims.",
        ]
        if quant_ratio < 0.3:
            risks.append("Low measurable outcome density may weaken confidence.")
        if repeat_ratio > 0.12:
            risks.append("Repetitive bullets reduce clarity of unique contributions.")

        evidence = _serialize_examples(_find_snippets(resume, list(jd_terms)[:8], limit=3))
        interview_focus = [
            "Ownership and trade-off decisions",
            "Metric baselines and verification method",
            "Role-critical system depth",
            "Failure handling and iteration",
        ]

        notes.append(
            ShortlistCandidateNote(
                label=candidate.candidate_label,
                fit_level=fit_level,
                top_strengths=strengths[:7],
                top_risks=risks[:7],
                evidence_snippets=evidence,
                interview_focus=interview_focus[:7],
            )
        )
        scored.append((candidate.candidate_label, score))

    scored.sort(key=lambda row: row[1], reverse=True)
    shortlist = [label for label, _ in scored[:2]]
    recommendation = (
        f"Shortlist {', '.join(shortlist)} for structured interviews, then compare against role-critical gaps."
        if shortlist
        else "Shortlist decision is unclear; gather more evidence before narrowing."
    )

    decision_summary = (
        "The shortlist recommendation balances role-term alignment, evidence quality, and interview risk. "
        "Use these notes to keep evaluation criteria consistent across interviewers."
    )

    notes_lines = [f"Shortlist recommendation: {recommendation}", "", "Candidate notes:"]
    for item in notes:
        notes_lines.append(f"- {item.label}: {item.fit_level} fit.")
        notes_lines.append(f"  Strengths: {', '.join(item.top_strengths[:3])}")
        notes_lines.append(f"  Risks: {', '.join(item.top_risks[:3])}")
    copyable = "\n".join(notes_lines).strip()

    return ShortlistJustificationResponse(
        shortlist_recommendation=recommendation,
        decision_summary=decision_summary,
        candidate_notes=notes,
        copyable_hiring_notes=copyable,
        generated_at=_utc_now(),
    )


def run_shortlist_justification(payload: ShortlistJustificationRequest) -> ShortlistJustificationResponse:
    fallback = _shortlist_justification_fallback(payload)
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You assist recruiters with shortlist justification. Keep output evidence-based and avoid absolute winner claims. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"JD:\n{payload.jd_text[:9000]}\n\n"
            f"Candidates:\n{[{'label': item.candidate_label, 'resume_text': item.resume_text[:6000]} for item in payload.candidates]}\n\n"
            "Return JSON keys: shortlist_recommendation, decision_summary, candidate_notes, copyable_hiring_notes.\n"
            "candidate_notes items: label, fit_level(Strong/Moderate/Weak), top_strengths, top_risks, evidence_snippets, interview_focus."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return ShortlistJustificationResponse(
            shortlist_recommendation=str(llm_payload.get("shortlist_recommendation", fallback.shortlist_recommendation)),
            decision_summary=str(llm_payload.get("decision_summary", fallback.decision_summary)),
            candidate_notes=llm_payload.get(
                "candidate_notes",
                [note.model_dump() for note in fallback.candidate_notes],
            ),
            copyable_hiring_notes=str(llm_payload.get("copyable_hiring_notes", fallback.copyable_hiring_notes)),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Shortlist justification output failed strict validation.", status_code=422) from exc
        return fallback


def _hiring_bias_risk_fallback(payload: HiringBiasRiskDetectorRequest) -> HiringBiasRiskDetectorResponse:
    source_text = "\n".join([payload.jd_text.strip(), payload.evaluation_text.strip()]).strip()
    lower = source_text.lower()

    flagged: list[BiasRiskFlag] = []
    for category, patterns in _BIAS_PATTERNS.items():
        for phrase in patterns:
            if phrase in lower:
                flagged.append(
                    BiasRiskFlag(
                        phrase=phrase,
                        category=category,
                        why_it_matters="This wording can reduce objectivity and discourage qualified candidates.",
                        safer_alternative=f"Replace with role-specific, measurable criteria instead of '{phrase}'.",
                    )
                )
            if len(flagged) >= 14:
                break
        if len(flagged) >= 14:
            break

    clarity_improvements = [
        "Define must-have capabilities as measurable outcomes rather than personality labels.",
        "Use role-relevant behaviors and evidence criteria for interview scoring.",
        "Separate mandatory requirements from preferred qualifications.",
        "Replace vague culture-fit wording with concrete collaboration expectations.",
    ]
    if payload.evaluation_text.strip():
        clarity_improvements.append("Anchor evaluation notes to observed evidence from interviews or work samples.")

    risk_score = min(1.0, len(flagged) / 6.0)
    if any(item.category in {"visa bias", "age-coded", "ableist"} for item in flagged):
        risk_score = min(1.0, risk_score + 0.2)
    risk_level = _risk_from_score(risk_score)

    summary = (
        "Bias-risk signals are limited, but wording can still be tightened for consistent, evidence-based screening."
    )
    if risk_level == "Medium":
        summary = (
            "Several wording patterns may increase bias risk or reduce candidate trust. "
            "Refine phrasing to focus on objective role outcomes."
        )
    elif risk_level == "High":
        summary = (
            "Multiple high-impact wording patterns indicate elevated bias risk. "
            "Prioritize neutral, measurable language before publishing or finalizing notes."
        )

    if not flagged:
        flagged.append(
            BiasRiskFlag(
                phrase="No high-risk phrase detected",
                category="culture-fit vague",
                why_it_matters="A clean scan still benefits from objective, role-specific scoring criteria.",
                safer_alternative="Continue using structured, evidence-based evaluation language.",
            )
        )

    return HiringBiasRiskDetectorResponse(
        bias_risk_level=risk_level,
        summary=summary,
        flagged_phrases=flagged[:14],
        clarity_improvements=clarity_improvements[:8],
        generated_at=_utc_now(),
    )


def run_hiring_bias_risk_detector(payload: HiringBiasRiskDetectorRequest) -> HiringBiasRiskDetectorResponse:
    fallback = _hiring_bias_risk_fallback(payload)
    source_text = "\n".join([payload.jd_text.strip(), payload.evaluation_text.strip()]).strip()
    llm_payload = _llm_json_with_policy(
        system_prompt=(
            "You assist recruiters in reducing wording bias. Be neutral and non-moralizing. "
            "Do not give legal advice. Return strict JSON only."
        ),
        user_prompt=(
            f"Analyze this text for hiring bias wording risk:\n{source_text[:14000]}\n\n"
            "Return JSON keys: bias_risk_level, summary, flagged_phrases, clarity_improvements.\n"
            "flagged_phrases items: phrase, category(one of gender-coded, age-coded, culture-fit vague, ableist, aggressive tone, elitism/credential bias, location bias, visa bias), why_it_matters, safer_alternative."
        ),
    )
    if not llm_payload:
        return fallback
    try:
        return HiringBiasRiskDetectorResponse(
            bias_risk_level=llm_payload.get("bias_risk_level", fallback.bias_risk_level),
            summary=str(llm_payload.get("summary", fallback.summary)),
            flagged_phrases=llm_payload.get("flagged_phrases", [item.model_dump() for item in fallback.flagged_phrases]),
            clarity_improvements=llm_payload.get("clarity_improvements", fallback.clarity_improvements),
            generated_at=_utc_now(),
        )
    except Exception as exc:
        if _is_strict_mode():
            raise RecruiterQualityError("Bias risk detector output failed strict quality validation.", status_code=422) from exc
        return fallback


def validate_share_payload(payload: RecruiterShareCreateRequest) -> None:
    allowed = {
        "resume-authenticity",
        "claim-verification",
        "jd-quality",
        "ats-vs-human",
        "resume-compare",
        "resume-signal-strength",
        "jd-market-reality",
        "role-seniority-definition",
        "shortlist-justification",
        "bias-risk-detector",
    }
    if payload.tool_slug not in allowed:
        raise RecruiterQualityError("Unknown recruiter tool for share link.", status_code=400)
