from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from portfolio_agent.errors import ToolError
from portfolio_agent.tools.http import get_json


@dataclass(frozen=True)
class CrossrefWork:
    title: str
    url: str
    year: int | None
    snippet: str


def crossref_search(query: str, *, rows: int = 5) -> list[CrossrefWork]:
    safe = quote(query)
    url = f"https://api.crossref.org/works?query={safe}&rows={rows}"
    resp = get_json(url, headers={"User-Agent": "agentic-brief/0.1"})

    if resp.status_code >= 400:
        raise ToolError(
            f"Crossref error {resp.status_code} for '{query}'",
            tool="crossref_search",
            kind="http_error",
            status_code=resp.status_code,
        )

    items = (((resp.json.get("message") or {}).get("items")) or [])[:rows]
    works: list[CrossrefWork] = []
    for it in items:
        title = ""
        if isinstance(it.get("title"), list) and it.get("title"):
            title = str(it["title"][0])
        url_ = str(it.get("URL") or "")
        issued = ((it.get("issued") or {}).get("date-parts") or [])
        year = None
        if issued and isinstance(issued[0], list) and issued[0]:
            try:
                year = int(issued[0][0])
            except Exception:
                year = None
        container = ""
        if isinstance(it.get("container-title"), list) and it.get("container-title"):
            container = str(it["container-title"][0])
        snippet_parts = [p for p in [container, str(year) if year else ""] if p]
        snippet = " • ".join(snippet_parts).strip()

        if title and url_:
            works.append(CrossrefWork(title=title, url=url_, year=year, snippet=snippet))

    return works
