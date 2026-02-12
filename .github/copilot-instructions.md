# Copilot/Codex Repository Rules

- Do not analyze raw resume/JD text directly; use normalized schema objects only.
- Route all LLM calls through the shared safe wrapper with strict JSON and Pydantic validation.
- Every analytical finding must include evidence references (`claim_id`, `req_id`, or explicit evidence span).
- Never invent facts; if evidence is missing, return safe fallback behavior and request user input when needed.
- Implement deterministic fallbacks for tool outputs when model output is invalid or low-confidence.
- Keep scoring thresholds/weights config-driven from `config/scoring.yaml` (no hardcoded tool scoring constants).
- Do not log raw resume/JD content or PII; redact sensitive fields by default.

