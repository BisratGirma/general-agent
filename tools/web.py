"""Web search and HTML scraping tools."""

from __future__ import annotations

import os

import requests

from tools.common import _coerce_text, _format_error, _normalize_query

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

_BLOCKED_URL_FRAGMENTS = (
    "youtube.com/watch",
    "youtu.be/",
    "vimeo.com/",
    "tiktok.com/",
    "instagram.com/",
    "facebook.com/",
    "twitter.com/",
    "soundcloud.com/",
)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _is_text_webpage_url(url: str) -> bool:
    """Return ``True`` when *url* points to a text-based page (not video/audio/social)."""
    url = (url or "").lower()
    return not any(block in url for block in _BLOCKED_URL_FRAGMENTS)


def _is_wikipedia_query(query: str) -> bool:
    """Return ``True`` when the query explicitly mentions Wikipedia or Wikimedia."""
    normalized = (query or "").lower()
    return "wikipedia" in normalized or "wikimedia" in normalized


# ---------------------------------------------------------------------------
# Page-selection helpers
# ---------------------------------------------------------------------------

def _build_page_selection_prompt(user_query: str, results: list[dict]) -> str:
    """Build a prompt that asks an LLM to choose the best URL from *results*."""
    lines = [
        f"user requested this: {user_query}",
        "",
        "from the list",
    ]
    for result in results:
        title = result.get("title", "Untitled") or "Untitled"
        url = result.get("url") or result.get("href") or result.get("link") or ""
        snippet = (result.get("snippet") or result.get("body") or "").strip()
        if not url:
            continue
        line = f"- {title} / {url}"
        if snippet:
            line += f" / snippet: {snippet}"
        lines.append(line)
    lines.extend([
        "",
        "Choose the single website link (page) that most directly contains the information the user requested.",
        "Prefer text-based pages such as Wikipedia, news articles, or encyclopedia pages.",
        "Avoid selecting video pages, streaming pages, audio-only content, or social media posts.",
        "If a Wikipedia page is available, prefer it. Otherwise choose the best text-based page.",
        "Respond with only the chosen URL.",
    ])
    return "\n".join(lines)


def _build_llm_client():
    """Return an OpenAI-compatible client if credentials are available, else None."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not base_url and not api_key:
        return None

    try:
        if base_url:
            return OpenAI(base_url=base_url, api_key=api_key or "openai")
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _select_best_page_from_results(user_query: str, results: list[dict]) -> str:
    """Pick the single best URL from *results* for *user_query*.

    Priority order:
    1. Wikipedia URL (when query mentions Wikipedia).
    2. LLM-selected URL (when an LLM client is available).
    3. First valid text-based URL.
    """
    valid_pages = [
        {
            "title": r.get("title", "Untitled"),
            "url": u,
            "snippet": r.get("snippet", r.get("body", "")),
        }
        for r in results
        if (u := r.get("url") or r.get("href") or r.get("link") or "")
        and u.startswith(("http://", "https://"))
    ]

    if not valid_pages:
        return ""

    text_pages = [p for p in valid_pages if _is_text_webpage_url(p["url"])] or valid_pages

    # Fast path: Wikipedia requested
    if _is_wikipedia_query(user_query):
        for page in text_pages:
            if "wikipedia.org" in page["url"].lower():
                return page["url"]

    # LLM path
    client = _build_llm_client()
    if client is None:
        return text_pages[0]["url"]

    prompt = _build_page_selection_prompt(user_query, text_pages)
    model_name = os.getenv("WEB_SEARCH_SELECTION_MODEL", "gpt-3.5-turbo")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.1,
            top_p=1.0,
            stream=False,
        )
        selected = _coerce_text(
            response.choices[0].message.content
            if hasattr(response, "choices") and response.choices
            else response
        ).strip()
        selected_url = selected.strip().split()[0] if selected else ""
        if selected_url.startswith(("http://", "https://")) and _is_text_webpage_url(selected_url):
            return selected_url
    except Exception:
        pass

    return text_pages[0]["url"]


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def scrape_url(url: str) -> str:
    """Fetch a web page and convert it to readable plain text."""
    url = _normalize_query(url)
    if not url.startswith(("http://", "https://")):
        return _format_error("HTML Scrape", "URL must start with http:// or https://")

    try:
        response = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return _format_error("HTML Scrape", str(exc))

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _format_error("HTML Scrape", "beautifulsoup4 is not installed")

    soup = BeautifulSoup(response.text, "html.parser")
    text_parts = [
        " ".join(tag.get_text(" ", strip=True).split())
        for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
    ]
    text = "\n".join(filter(None, text_parts))[:6000]
    return text or _format_error("HTML Scrape", "no readable text found")


def web_search(query: str, max_results: int = 10) -> str:
    """Search DuckDuckGo, pick the best result page, and return scraped content."""
    query = _normalize_query(query)
    if not query:
        return _format_error("Web Search", "empty query")

    try:
        from ddgs import DDGS
    except ImportError:
        return _format_error("Web Search", "ddgs package is not installed")

    try:
        with DDGS() as ddgs:
            print(f"Performing web search for: {query}")
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return _format_error("Web Search", str(exc))

    if not raw_results:
        return "No web search results found."

    output = [
        {"title": r.get("title", "Untitled"), "url": r.get("href") or r.get("link", ""), "snippet": r.get("body", "")}
        for r in raw_results
        if r.get("href") or r.get("link")
    ]

    if not output:
        return "No processable URLs found."

    selected_url = _select_best_page_from_results(query, output)
    if not selected_url:
        return "No valid page could be selected from search results."

    print(f"Selected page to scrape: {selected_url}")
    scraped = scrape_url(selected_url)
    return f"=== Selected Source ===\nURL: {selected_url}\nContent:\n{scraped}"
