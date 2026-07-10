from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class Source(TypedDict):
    api: Literal["wikipedia", "crossref"]
    title: str
    url: str
    snippet: str
    meta: NotRequired[dict[str, Any]]


class ToolCall(TypedDict):
    tool: str
    args: dict[str, Any]


class LastError(TypedDict, total=False):
    tool: str
    kind: str
    status_code: int | None
    message: str


class AgentState(TypedDict, total=False):
    thread_id: str

    topic: str
    audience: str
    constraints: str
    refine_only: bool
    fail_demo: bool
    offline_tools: bool

    research_questions: list[str]
    tool_plan: list[ToolCall]
    sources: list[Source]
    notes: list[str]

    draft: str
    final: str

    loop_count: int
    needs_more_research: bool
    last_error: LastError | None
