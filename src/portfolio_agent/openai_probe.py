from __future__ import annotations

import sys

from portfolio_agent.config import load_config
from portfolio_agent.llm import get_llm_with_mode


def main() -> None:
    cfg = load_config()

    llm, mode = get_llm_with_mode(cfg.openai_api_key)

    expected = "openai" if cfg.openai_api_key else "rule_based"
    if mode != expected:
        print(f"probe: expected mode={expected} got mode={mode}", file=sys.stderr)
        sys.exit(2)

    if mode == "openai":
        try:
            _ = llm.plan(topic="Tool calling", audience="probe", constraints="Return minimal JSON.")
        except Exception as e:
            print(f"probe: openai call failed: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(3)

    print(f"probe: ok (mode={mode})")


if __name__ == "__main__":
    main()
