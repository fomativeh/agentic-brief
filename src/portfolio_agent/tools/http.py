from __future__ import annotations

from dataclasses import dataclass

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from portfolio_agent.errors import ToolError


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    json: dict


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception(
        lambda e: isinstance(e, ToolError) and e.kind in {"network_error", "retryable_http"}
    ),
)
def get_json(url: str, *, timeout_s: float = 15.0, headers: dict[str, str] | None = None) -> HttpResponse:
    try:
        resp = requests.get(url, timeout=timeout_s, headers=headers)
    except requests.RequestException as e:
        raise ToolError(str(e), tool="http_get_json", kind="network_error") from e

    if resp.status_code == 429 or resp.status_code >= 500:
        raise ToolError(
            f"Retryable HTTP {resp.status_code} from {url}",
            tool="http_get_json",
            kind="retryable_http",
            status_code=resp.status_code,
        )

    try:
        data = resp.json()
    except ValueError as e:
        raise ToolError(
            f"Non-JSON response from {url} (status {resp.status_code})",
            tool="http_get_json",
            kind="bad_response",
            status_code=resp.status_code,
        ) from e

    if not isinstance(data, dict):
        raise ToolError(
            f"Unexpected JSON type from {url}: {type(data).__name__}",
            tool="http_get_json",
            kind="bad_response",
            status_code=resp.status_code,
        )

    return HttpResponse(status_code=resp.status_code, json=data)
