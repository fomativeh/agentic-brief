from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from portfolio_agent.graph import build_app


def main() -> None:
    thread = f"smoke-{uuid.uuid4().hex[:8]}"
    online = os.getenv("AGENTIC_BRIEF_SMOKE_ONLINE") == "1"

    with tempfile.TemporaryDirectory() as td:
        sqlite_path = str(Path(td) / "state.sqlite3")
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(sqlite_path) as saver:
            app = build_app(openai_api_key=None, checkpointer=saver)

            run1 = app.invoke(
                {
                    "topic": "Retrieval-augmented generation",
                    "constraints": "One page max.",
                    "fail_demo": False,
                    "offline_tools": not online,
                },
                config={"configurable": {"thread_id": thread}},
            )
            sources1 = list(run1.get("sources") or [])
            notes1 = list(run1.get("notes") or [])
            if not sources1:
                raise SystemExit("smoke: expected sources after run")
            if not any(n == "llm: rule_based" for n in notes1):
                raise SystemExit("smoke: expected llm mode note")

            snap = app.get_state({"configurable": {"thread_id": thread}})
            values = getattr(snap, "values", None) or getattr(snap, "state", None) or {}
            if not (values.get("topic") and values.get("sources")):
                raise SystemExit("smoke: expected persisted topic + sources in checkpointed state")

            run2 = app.invoke(
                {"constraints": "Under 200 words.", "refine_only": True},
                config={"configurable": {"thread_id": thread}},
            )
            sources2 = list(run2.get("sources") or [])
            if len(sources2) != len(sources1):
                raise SystemExit("smoke: expected refine to reuse sources without changing count")

            run3 = app.invoke(
                {"fail_demo": True, "refine_only": False, "offline_tools": not online},
                config={"configurable": {"thread_id": thread}},
            )
            notes3 = list(run3.get("notes") or [])
            if not any(str(n).startswith("fail_demo:") for n in notes3):
                raise SystemExit("smoke: expected fail-demo note recorded in state")

    print("smoke: ok")


if __name__ == "__main__":
    main()
