from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from portfolio_agent.errors import ToolError
from portfolio_agent.llm import count_words, extract_word_cap, get_llm, get_llm_with_mode, truncate_words
from portfolio_agent.state import AgentState, Source, ToolCall
from portfolio_agent.tools.crossref import crossref_search
from portfolio_agent.tools.wikipedia import wikipedia_search, wikipedia_summary


MAX_RESEARCH_LOOPS = 2


def _ensure_defaults(state: AgentState) -> AgentState:
    state.setdefault("audience", "prompt engineer")
    state.setdefault("constraints", "")
    state.setdefault("research_questions", [])
    state.setdefault("tool_plan", [])
    state.setdefault("sources", [])
    state.setdefault("notes", [])
    state.setdefault("loop_count", 0)
    state.setdefault("needs_more_research", False)
    state.setdefault("last_error", None)
    state.setdefault("refine_only", False)
    state.setdefault("fail_demo", False)
    state.setdefault("offline_tools", False)
    return state


def _pick_best_title(query: str, candidates: list[str]) -> str:
    q = query.strip().lower()
    if not q:
        return candidates[0]
    for c in candidates:
        if c.strip().lower() == q:
            return c
    for c in candidates:
        if q in c.strip().lower():
            return c
    return candidates[0]


def intake_node(state: AgentState) -> dict[str, Any]:
    state = _ensure_defaults(state)
    if "topic" not in state or not str(state.get("topic") or "").strip():
        raise ValueError("Missing required 'topic'.")
    return {"topic": str(state["topic"]).strip(), "last_error": None}


def _set_last_error(state: AgentState, *, tool: str, kind: str, message: str, status_code: int | None = None) -> None:
    state["last_error"] = {"tool": tool, "kind": kind, "status_code": status_code, "message": message}


def planner_node(state: AgentState, *, openai_api_key: str | None) -> dict[str, Any]:
    state = _ensure_defaults(state)
    llm, llm_mode = get_llm_with_mode(openai_api_key)
    if not any(str(n).startswith("llm:") for n in (state.get("notes") or [])):
        state["notes"].append(f"llm: {llm_mode}")

    if state.get("refine_only") and state.get("sources"):
        return {"notes": state["notes"]}

    plan = llm.plan(topic=state["topic"], audience=state["audience"], constraints=state["constraints"])
    tool_plan: list[ToolCall] = []
    for step in plan.tool_plan:
        tool = step.get("tool")
        args = step.get("args")
        if not isinstance(tool, str) or not tool.strip():
            state["notes"].append("tool_plan_error: missing tool name")
            _set_last_error(state, tool="planner", kind="tool_plan_error", message="missing tool name")
            continue

        if not isinstance(args, dict):
            args = {}

        if tool in {"wikipedia_summary", "wikipedia_search", "crossref_search"}:
            tool_plan.append({"tool": tool, "args": args})
            continue

        msg = f"unknown_tool: {tool}"
        if msg not in state["notes"]:
            state["notes"].append(msg)
        _set_last_error(state, tool="planner", kind="unknown_tool", message=tool)

    if not tool_plan:
        tool_plan = [
            {"tool": "wikipedia_summary", "args": {"title": state["topic"]}},
            {"tool": "crossref_search", "args": {"query": f'{state["topic"]} survey review', "rows": 5}},
        ]

    return {
        "research_questions": plan.research_questions,
        "tool_plan": tool_plan,
        "notes": state["notes"],
        "last_error": state.get("last_error"),
    }


def _add_source(state: AgentState, src: Source) -> None:
    existing_urls = {s.get("url") for s in state.get("sources") or []}
    if src["url"] not in existing_urls:
        state["sources"].append(src)


def research_node(state: AgentState) -> dict[str, Any]:
    state = _ensure_defaults(state)
    if state.get("refine_only") and state.get("sources"):
        return {}

    state["last_error"] = None
    had_error = False

    tool_plan = state.get("tool_plan") or []

    if state.get("fail_demo") and not any(n.startswith("fail_demo:") for n in state.get("notes") or []):
        if state.get("offline_tools"):
            state["notes"].append("fail_demo: simulated wikipedia_summary not_found 404")
        else:
            try:
                wikipedia_summary("this_page_should_not_exist_abcdefg_12345")
            except ToolError as e:
                state["notes"].append(f"fail_demo: triggered {e.tool} {e.kind} {e.status_code}")

    for step in tool_plan:
        tool = step["tool"]
        args = step.get("args") or {}

        try:
            if tool == "wikipedia_summary":
                title = str(args.get("title") or state["topic"])
                if state.get("offline_tools"):
                    _add_source(
                        state,
                        {
                            "api": "wikipedia",
                            "title": title,
                            "url": f"https://example.local/wiki/{title.replace(' ', '_')}",
                            "snippet": f"Offline fixture summary for {title}.",
                        },
                    )
                    continue
                try:
                    ws = wikipedia_summary(title)
                except ToolError as e:
                    if e.kind == "not_found":
                        candidates = wikipedia_search(title, limit=5)
                        if not candidates:
                            raise
                        best = _pick_best_title(title, candidates)
                        ws = wikipedia_summary(best)
                        state["notes"].append(f"recovered: wikipedia_summary fallback '{title}' -> '{best}'")
                    else:
                        raise

                _add_source(
                    state,
                    {
                        "api": "wikipedia",
                        "title": ws.title,
                        "url": ws.url,
                        "snippet": ws.extract[:500].strip(),
                    },
                )

            elif tool == "wikipedia_search":
                query = str(args.get("query") or state["topic"])
                if state.get("offline_tools"):
                    state["notes"].append(f"candidate: {query}")
                    continue
                titles = wikipedia_search(query, limit=int(args.get("limit") or 5))
                for t in titles[:3]:
                    state["notes"].append(f"candidate: {t}")

            elif tool == "crossref_search":
                query = str(args.get("query") or state["topic"])
                rows = int(args.get("rows") or 5)
                if state.get("offline_tools"):
                    _add_source(
                        state,
                        {
                            "api": "crossref",
                            "title": f"Offline fixture paper about {query}",
                            "url": "https://example.local/paper",
                            "snippet": "Offline fixture",
                            "meta": {"year": 2026},
                        },
                    )
                    continue
                works = crossref_search(query, rows=rows)
                for w in works:
                    _add_source(
                        state,
                        {
                            "api": "crossref",
                            "title": w.title,
                            "url": w.url,
                            "snippet": w.snippet,
                            "meta": {"year": w.year},
                        },
                    )

            else:
                msg = f"unknown_tool: {tool}"
                if msg not in state["notes"]:
                    state["notes"].append(msg)
                had_error = True
                _set_last_error(state, tool="research", kind="unknown_tool", message=str(tool))

        except ToolError as e:
            had_error = True
            state["last_error"] = e.to_error()
            state["notes"].append(f"tool_error: {e.tool}:{e.kind}:{e.status_code}")

    if not had_error:
        state["last_error"] = None

    return {"sources": state["sources"], "notes": state["notes"], "last_error": state.get("last_error")}


def synthesizer_node(state: AgentState, *, openai_api_key: str | None) -> dict[str, Any]:
    state = _ensure_defaults(state)
    llm = get_llm(openai_api_key)
    draft = llm.synthesize(
        topic=state["topic"],
        audience=state["audience"],
        constraints=state["constraints"],
        sources=state.get("sources") or [],
    )
    return {"draft": draft}


def critic_node(state: AgentState, *, openai_api_key: str | None) -> dict[str, Any]:
    state = _ensure_defaults(state)
    llm = get_llm(openai_api_key)
    critique = llm.critique(draft=state.get("draft") or "", sources=state.get("sources") or [])

    needs_more = bool(critique.needs_more_research) and state.get("loop_count", 0) < MAX_RESEARCH_LOOPS
    update: dict[str, Any] = {
        "needs_more_research": needs_more,
        "loop_count": int(state.get("loop_count") or 0) + (1 if needs_more else 0),
    }

    if needs_more and critique.suggested_queries:
        q = critique.suggested_queries[0]
        state["tool_plan"] = list(state.get("tool_plan") or []) + [
            {"tool": "crossref_search", "args": {"query": f"{state['topic']} {q}", "rows": 5}}
        ]
        update["tool_plan"] = state["tool_plan"]
        state["notes"].append(f"critic: requested more research ({q})")
        update["notes"] = state["notes"]

    return update


def finalize_node(state: AgentState) -> dict[str, Any]:
    state = _ensure_defaults(state)
    final = state.get("draft") or ""
    cap = extract_word_cap(state.get("constraints") or "")
    if cap is not None:
        final = truncate_words(final, cap)
        wc = count_words(final)
        note = f"word_cap:{cap} final_words:{wc}"
        if note not in state["notes"]:
            state["notes"].append(note)
        return {"final": final, "notes": state["notes"]}
    return {"final": final}


def _route_after_critic(state: AgentState) -> str:
    return "research" if state.get("needs_more_research") else "finalize"


def build_app(*, sqlite_path: str | None = None, openai_api_key: str | None, checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("planner", lambda s: planner_node(s, openai_api_key=openai_api_key))
    graph.add_node("research", research_node)
    graph.add_node("synthesizer", lambda s: synthesizer_node(s, openai_api_key=openai_api_key))
    graph.add_node("critic", lambda s: critic_node(s, openai_api_key=openai_api_key))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, {"research": "research", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    if checkpointer is None:
        if not sqlite_path:
            raise ValueError("Either sqlite_path or checkpointer is required.")
        checkpointer = _make_sqlite_checkpointer(str(sqlite_path))

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


def _make_sqlite_checkpointer(path: str):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as e:
        raise RuntimeError("Missing langgraph-checkpoint-sqlite dependency.") from e

    import sqlite3

    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)
