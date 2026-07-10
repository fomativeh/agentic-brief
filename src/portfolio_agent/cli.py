from __future__ import annotations

import argparse
import json
import sys

from portfolio_agent.config import load_config
from portfolio_agent.graph import build_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentic-brief")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the full workflow (intake -> plan -> tools -> write -> critique).")
    run_p.add_argument("--thread", default="default")
    run_p.add_argument("--topic", required=True)
    run_p.add_argument("--audience", default="prompt engineer")
    run_p.add_argument("--constraints", default="")
    run_p.add_argument("--fail-demo", action="store_true")

    refine_p = sub.add_parser("refine", help="Reuse prior state and regenerate the brief with new constraints.")
    refine_p.add_argument("--thread", default="default")
    refine_p.add_argument("--constraints", required=True)
    refine_p.add_argument("--audience", default=None)

    state_p = sub.add_parser("state", help="Print the current stored state snapshot for a thread.")
    state_p.add_argument("--thread", default="default")

    args = parser.parse_args()

    cfg = load_config()
    app = build_app(sqlite_path=str(cfg.sqlite_path), openai_api_key=cfg.openai_api_key)

    if args.cmd == "run":
        payload = {
            "topic": args.topic,
            "audience": args.audience,
            "constraints": args.constraints,
            "refine_only": False,
            "fail_demo": bool(args.fail_demo),
        }
        result = app.invoke(payload, config={"configurable": {"thread_id": args.thread}})
        _print_result(result)
        return

    if args.cmd == "refine":
        payload = {"constraints": args.constraints, "refine_only": True}
        if args.audience is not None:
            payload["audience"] = args.audience
        result = app.invoke(payload, config={"configurable": {"thread_id": args.thread}})
        _print_result(result)
        return

    if args.cmd == "state":
        config = {"configurable": {"thread_id": args.thread}}
        if not hasattr(app, "get_state"):
            print("This version of LangGraph does not expose get_state().", file=sys.stderr)
            sys.exit(2)
        snap = app.get_state(config)
        values = getattr(snap, "values", None) or getattr(snap, "state", None) or {}
        print(json.dumps(values, indent=2, ensure_ascii=False))
        return


def _print_result(result: dict) -> None:
    final = result.get("final") or result.get("draft") or ""
    notes = result.get("notes") or []
    last_error = result.get("last_error")

    if notes:
        print("== Notes ==")
        for n in notes[-10:]:
            print(f"- {n}")
        print()

    if last_error:
        print("== Last Error ==")
        if isinstance(last_error, dict):
            tool = last_error.get("tool")
            kind = last_error.get("kind")
            code = last_error.get("status_code")
            msg = last_error.get("message")
            label = f"{tool}:{kind}:{code}" if tool or kind or code is not None else "error"
            print(label)
            if msg:
                print(msg)
        else:
            print(last_error)
        print()

    print(final)


if __name__ == "__main__":
    main()
