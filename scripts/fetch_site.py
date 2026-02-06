from __future__ import annotations

import argparse
import re
from pathlib import Path
from html import unescape

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None

import urllib.request
try:
    from playwright.sync_api import sync_playwright  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sync_playwright = None


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _needs_render(text: str) -> bool:
    if not text:
        return True
    if "Loading" in text and len(text) < 500:
        return True
    return len(text) < 400


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch website text into a markdown file.")
    parser.add_argument("--url", default="https://www.codedbyasad.com", help="Website URL")
    parser.add_argument(
        "--out",
        default="data/documents/en/site.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Force JS rendering using Playwright (if installed).",
    )
    args = parser.parse_args()

    html = ""
    if httpx is not None:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(args.url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
    else:
        with urllib.request.urlopen(args.url, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")

    text = _strip_html(html)

    if args.render or _needs_render(text):
        if sync_playwright is None:
            raise RuntimeError(
                "Page appears to be JS-rendered. Install Playwright to render it:\n"
                "  pip install playwright\n"
                "  python -m playwright install\n"
                "Then rerun with --render."
            )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(args.url, wait_until="networkidle", timeout=30_000)
            html = page.content()
            browser.close()
        text = _strip_html(html)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# Website Snapshot\n\nSource: {args.url}\n\n{text}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
