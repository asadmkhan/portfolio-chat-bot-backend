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
    ClaimVerificationRequest,
    ClaimVerificationResponse,
    JDQualityAnalyzerRequest,
    JDQualityIssue,
    JDQualityResponse,
    QuickWin,
    RecruiterShareCreateRequest,
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


def validate_share_payload(payload: RecruiterShareCreateRequest) -> None:
    allowed = {
        "resume-authenticity",
        "claim-verification",
        "jd-quality",
        "ats-vs-human",
        "resume-compare",
    }
    if payload.tool_slug not in allowed:
        raise RecruiterQualityError("Unknown recruiter tool for share link.", status_code=400)
