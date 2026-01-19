"""Web, RSS, scraping, and sandboxed notebook tools."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import statistics
from html.parser import HTMLParser
from typing import List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:  # pragma: no cover - optional dependency
    from agent.web.web_search import WebSearchUnavailable, search_web
except Exception:  # pragma: no cover - optional dependency
    WebSearchUnavailable = None  # type: ignore[assignment]
    search_web = None  # type: ignore[assignment]


RSS_URLS_ENV = "NEWS_RSS_URLS"
DEFAULT_RSS_URLS = ["https://feeds.reuters.com/reuters/topNews"]
BLUESKY_API_URL = os.environ.get("BLUESKY_API_URL", "https://public.api.bsky.app")
BLUESKY_AUTH_API_URL = os.environ.get("BLUESKY_AUTH_API_URL", "https://bsky.social")
BLUESKY_AUTH_TOKEN = os.environ.get("BLUESKY_AUTH_TOKEN")
BLUESKY_USER_AGENT = os.environ.get("BLUESKY_USER_AGENT", "AgentTimeBot/1.0")


def web_search_available() -> bool:
    return search_web is not None


def _summarize_search_results(results: List[object]) -> str:
    if not results:
        return "No results."
    lines = [f"Results: {len(results)} found."]
    for idx, result in enumerate(results[:3], 1):
        title = getattr(result, "title", "Untitled result")
        url = getattr(result, "url", "")
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"   {url}")
    if len(results) > 3:
        lines.append(f"... {len(results) - 3} more.")
    return "\n".join(lines)


def _run_search(query: str, limit: int = 5) -> str:
    if search_web is None:
        raise RuntimeError("Web search tool unavailable. Install duckduckgo_search to enable it.")
    try:
        results = search_web(query, max_results=limit)
    except WebSearchUnavailable as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(str(exc)) from exc
    return _summarize_search_results(results)


def _load_rss_sources(sources: Optional[str | List[str]]) -> List[str]:
    if sources:
        if isinstance(sources, str):
            return [entry.strip() for entry in sources.split(",") if entry.strip()]
        return [source for source in sources if isinstance(source, str)]
    env_value = os.environ.get(RSS_URLS_ENV, "")
    sources_from_env = [entry.strip() for entry in env_value.split(",") if entry.strip()]
    return sources_from_env or DEFAULT_RSS_URLS


def _parse_rss_feed(xml_bytes: bytes) -> List[dict]:
    root = ET.fromstring(xml_bytes)
    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or "Untitled"
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        description = item.findtext("description") or ""
        items.append(
            {
                "title": title.strip(),
                "link": link.strip(),
                "pub_date": pub_date.strip(),
                "description": description.strip(),
            }
        )
    return items


def _run_rss_fetch(
    query: Optional[str] = None,
    limit: int = 10,
    sources: Optional[str | List[str]] = None,
) -> str:
    feeds = _load_rss_sources(sources)
    if not feeds:
        raise RuntimeError("No RSS feeds configured. Set NEWS_RSS_URLS or pass sources=[].")
    items: List[dict] = []
    for feed in feeds:
        try:
            with urllib.request.urlopen(feed, timeout=10) as response:
                xml_bytes = response.read()
            items.extend(_parse_rss_feed(xml_bytes))
        except Exception:
            continue
    if query:
        needle = query.lower()
        items = [
            item
            for item in items
            if needle in item.get("title", "").lower() or needle in item.get("description", "").lower()
        ]
    if not items:
        return "No results."
    lines = []
    for idx, item in enumerate(items[:limit], 1):
        lines.append(f"{idx}. {item.get('title')}")
        if item.get("link"):
            lines.append(f"   {item.get('link')}")
        if item.get("pub_date"):
            lines.append(f"   {item.get('pub_date')}")
    return "\n".join(lines)


def _run_bluesky_search(query: str, limit: int = 10) -> str:
    base_url = BLUESKY_AUTH_API_URL if BLUESKY_AUTH_TOKEN else BLUESKY_API_URL
    endpoint = f"{base_url.rstrip('/')}/xrpc/app.bsky.feed.searchPosts"
    params = {"q": query, "limit": limit}
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    headers = {
        "Accept": "application/json",
        "User-Agent": BLUESKY_USER_AGENT,
    }
    if BLUESKY_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {BLUESKY_AUTH_TOKEN}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    posts = payload.get("posts") if isinstance(payload, dict) else []
    if not isinstance(posts, list) or not posts:
        return "No results."
    lines = []
    for idx, post in enumerate(posts[:limit], 1):
        if not isinstance(post, dict):
            continue
        author = post.get("author", {})
        handle = ""
        if isinstance(author, dict):
            handle = author.get("handle") or ""
        record = post.get("record", {})
        text = ""
        if isinstance(record, dict):
            text = record.get("text") or ""
        line = f"{idx}. {text.strip()}"
        if handle:
            line += f" (@{handle})"
        lines.append(line)
        uri = post.get("uri")
        if uri:
            lines.append(f"   {uri}")
    return "\n".join(lines) if lines else "No results."


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_comment(self, data: str) -> None:
        # Drop comments to avoid analytics snippets
        return

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def get_text(self) -> str:
        return " ".join(self.parts)


def _run_web_scrape(url: str, max_chars: int = 2000) -> str:
    cleaned = (url or "").strip()
    if not cleaned.lower().startswith(("http://", "https://")):
        return f"Invalid URL. Provide a full http(s) URL. Got: {cleaned or '<empty>'}"
    try:
        request = urllib.request.Request(cleaned, headers={"User-Agent": "AgentTimeBot/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
    except Exception as exc:
        return f"Unable to fetch {cleaned}: {exc}"
    if "text/html" in content_type:
        parser = _HTMLTextExtractor()
        parser.feed(raw.decode("utf-8", errors="ignore"))
        text = parser.get_text()
    else:
        text = raw.decode("utf-8", errors="ignore")
    snippet = text[:max_chars].strip()
    return snippet or "No content found."


def _run_notebook_eval(code: str) -> str:
    safe_globals = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "print": print,
        },
        "math": math,
        "statistics": statistics,
    }
    safe_locals: dict = {}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, safe_locals)
    except Exception as exc:  # noqa: BLE001
        return f"Execution failed: {exc}"
    output = stdout.getvalue().strip()
    result = safe_locals.get("result")
    if result is not None and output:
        return f"Result: {result}\nOutput:\n{output}"
    if result is not None:
        return f"Result: {result}"
    return output or "No output."


__all__ = [
    "web_search_available",
    "_run_bluesky_search",
    "_run_notebook_eval",
    "_run_rss_fetch",
    "_run_search",
    "_run_web_scrape",
]
