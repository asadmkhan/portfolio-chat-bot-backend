import re

_DE_CHARS = re.compile(r"[äöüß]", re.IGNORECASE)
_DE_WORDS = re.compile(r"\b(und|ich|der|die|das|nicht|mit|für|ist|sind|habe|haben|wie|was|warum|bitte)\b", re.IGNORECASE)

def detect_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"
    if _DE_CHARS.search(t):
        return "de"
    if _DE_WORDS.search(t):
        return "de"
    return "en"
