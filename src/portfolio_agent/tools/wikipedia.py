from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from portfolio_agent.errors import ToolError
from portfolio_agent.tools.http import get_json


@dataclass(frozen=True)
class WikipediaSummary:
    title: str
    url: str
    extract: str


def wikipedia_summary(title: str) -> WikipediaSummary:
    safe = quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
    resp = get_json(url, headers={"User-Agent": "agentic-brief/0.1"})

    if resp.status_code == 404:
        raise ToolError(
            f"Wikipedia page not found for '{title}'",
            tool="wikipedia_summary",
            kind="not_found",
            status_code=404,
        )
    if resp.status_code >= 400:
        raise ToolError(
            f"Wikipedia summary error {resp.status_code} for '{title}'",
            tool="wikipedia_summary",
            kind="http_error",
            status_code=resp.status_code,
        )

    data = resp.json
    page_url = (data.get("content_urls", {}) or {}).get("desktop", {}).get("page")
    if not page_url:
        page_url = f"https://en.wikipedia.org/wiki/{safe}"

    return WikipediaSummary(
        title=data.get("title") or title,
        url=page_url,
        extract=data.get("extract") or "",
    )


def wikipedia_search(query: str, *, limit: int = 5) -> list[str]:
    safe = quote(query)
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={safe}&srlimit={limit}&format=json"
    )
    resp = get_json(url, headers={"User-Agent": "agentic-brief/0.1"})

    if resp.status_code >= 400:
        raise ToolError(
            f"Wikipedia search error {resp.status_code} for '{query}'",
            tool="wikipedia_search",
            kind="http_error",
            status_code=resp.status_code,
        )

    pages = ((resp.json.get("query") or {}).get("search")) or []
    titles: list[str] = []
    for p in pages:
        t = p.get("title")
        if isinstance(t, str) and t.strip():
            titles.append(t.strip())
    return titles
