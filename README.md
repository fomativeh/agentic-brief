# Agentic Brief (v1)

A small, resumable LangGraph workflow that calls real external APIs (Wikipedia + Crossref) and keeps state across runs via a SQLite checkpointer.

![](https://github.com/user-attachments/assets/19f40136-988c-4e97-b9b1-fe3ff6671d3e)


## What it does

Given a topic, it produces a short markdown “brief” for a target audience:

1. Intake (validate inputs, set defaults)
2. Plan (generate research questions + a minimal tool plan)
3. Research (execute tool plan against external APIs)
4. Synthesize (write the brief from gathered sources)
5. Critique (optionally request one more research pass, bounded)
6. Finalize (emit final markdown)

## State and resume

State is explicitly modeled (topic, constraints, tool plan, sources, notes, errors) and persisted across runs via a SQLite checkpointer keyed by a stable `thread_id`.

The workflow supports a deliberate failure mode (`--fail-demo`) that triggers a 404 from Wikipedia and continues gracefully, recording the incident in state.

## Non-goals

- No browser automation or HTML scraping. This stays CLI-first and API-first.
- No vector database RAG in this repo.
- No server or web UI. The CLI surface is the interface.
- No multi-user auth or isolation layer.
- Tool calling is kept loose; there is no deep schema enforcement.
- Tests are intentionally light (smoke test plus a few targeted checks).

## Implementation notes

- Typed state is defined up front; raw tool outputs stay in state and prompts are formatted on demand.
- Failures are represented in state (`last_error`, `notes`) and the workflow routes forward instead of crashing.
- Retries are bounded for transient HTTP failures (exponential backoff), and graph loops are bounded to avoid unbounded “self-improvement”.

## Related docs

- LangGraph: thinking in stateful graphs: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
- LangGraph fault tolerance / retries: https://docs.langchain.com/oss/python/langgraph/fault-tolerance.md

## Install

```bash
python -m venv .venv
.\.venv\Scripts\pip install -e .
```

## Optional: OpenAI mode

By default, this runs without any API keys (deterministic fallback). To use OpenAI, create a `.env` file (it is gitignored) and add:

```
OPENAI_API_KEY=...
```

Then install the optional dependency:

```bash
.\.venv\Scripts\pip install -c constraints-openai.txt -e ".[openai]"
```

Quick OpenAI probe:

```bash
.\.venv\Scripts\python -m portfolio_agent.openai_probe
```

## Run

```bash
.\.venv\Scripts\python -m portfolio_agent.cli run --thread demo --topic "Retrieval-augmented generation" --fail-demo
```

Run again with the same thread and new constraints (shows persisted state):

```bash
.\.venv\Scripts\python -m portfolio_agent.cli refine --thread demo --constraints "Keep it under 200 words."
```

## Smoke check

```bash
.\.venv\Scripts\python -m portfolio_agent.smoke_test
```

To force real external API calls:

```bash
$env:AGENTIC_BRIEF_SMOKE_ONLINE="1"
.\.venv\Scripts\python -m portfolio_agent.smoke_test
```
