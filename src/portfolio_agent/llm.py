from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel, Field

from portfolio_agent.state import ToolCall


class PlanResult(BaseModel):
    research_questions: list[str] = Field(min_length=1)
    tool_plan: list[dict[str, Any]] = Field(min_length=1)


class CritiqueResult(BaseModel):
    needs_more_research: bool
    suggested_queries: list[str] = Field(default_factory=list)


class LLMAdapter:
    def plan(self, *, topic: str, audience: str, constraints: str) -> PlanResult:
        raise NotImplementedError

    def synthesize(
        self,
        *,
        topic: str,
        audience: str,
        constraints: str,
        sources: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    def critique(self, *, draft: str, sources: list[dict[str, Any]]) -> CritiqueResult:
        raise NotImplementedError


@dataclass(frozen=True)
class RuleBasedLLM(LLMAdapter):
    def plan(self, *, topic: str, audience: str, constraints: str) -> PlanResult:
        questions = [
            f"What is {topic} and what problem does it solve?",
            f"What are the key terms and subtopics someone should know about {topic}?",
            f"What are common risks, limitations, or misunderstandings around {topic}?",
        ]
        tool_plan: list[ToolCall] = [
            {"tool": "wikipedia_summary", "args": {"title": topic}},
            {"tool": "crossref_search", "args": {"query": f'{topic} survey review', "rows": 5}},
        ]
        return PlanResult(research_questions=questions, tool_plan=tool_plan)

    def synthesize(
        self,
        *,
        topic: str,
        audience: str,
        constraints: str,
        sources: list[dict[str, Any]],
    ) -> str:
        wiki = next((s for s in sources if s.get("api") == "wikipedia"), None)
        papers = [s for s in sources if s.get("api") == "crossref"][:5]

        what_it_is = _shorten_text(str((wiki or {}).get("snippet") or ""), max_sentences=2, max_chars=320).strip()
        if not what_it_is:
            what_it_is = f"{topic} in one line: a technique or concept worth defining precisely before acting on it."

        paper_lines: list[str] = []
        for p in papers:
            paper_lines.append(f"- {p.get('title')} ({p.get('url')})")

        constraints_line = constraints.strip() if constraints.strip() else "None"
        draft = (
            f"# Brief: {topic}\n\n"
            f"Audience: {audience}\n\n"
            f"Constraints: {constraints_line}\n\n"
            f"## What it is\n"
            + f"- {what_it_is}\n\n"
            f"## Why it matters\n"
            f"- Practical relevance depends on the audience and their goals.\n"
            f"- The fast way to evaluate claims is to track definitions, assumptions, and evidence.\n\n"
            f"## Risks / misconceptions\n"
            f"- Hallucinated citations or fabricated facts if retrieval isn't verified.\n"
            f"- Stale or biased sources if your corpus isn't curated.\n"
            f"- Over-reliance on top-k chunks instead of checking primary sources.\n\n"
            f"## What to read next\n"
            + ("\n".join(paper_lines) if paper_lines else "- No papers found via Crossref for this query.\n")
            + "\n\n"
            f"## Sources\n"
            + "\n".join(f"- {s.get('title')} ({s.get('url')})" for s in sources[:10])
            + "\n"
        )

        word_cap = _extract_word_cap(constraints)
        if word_cap is not None:
            draft = _truncate_words(draft, word_cap)

        return draft

    def critique(self, *, draft: str, sources: list[dict[str, Any]]) -> CritiqueResult:
        if not sources:
            return CritiqueResult(needs_more_research=True, suggested_queries=["wikipedia", "crossref"])
        if len(draft.strip()) < 200:
            return CritiqueResult(needs_more_research=True, suggested_queries=["overview", "survey"])
        return CritiqueResult(needs_more_research=False, suggested_queries=[])


def get_llm(openai_api_key: str | None) -> LLMAdapter:
    if not openai_api_key:
        return RuleBasedLLM()

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        return RuleBasedLLM()

    class OpenAILLM(LLMAdapter):
        def __init__(self) -> None:
            self._planner = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            self._writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            self._critic = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        def plan(self, *, topic: str, audience: str, constraints: str) -> PlanResult:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a careful agent planner. Output a minimal tool plan.\n"
                        "Tools: wikipedia_summary(title), wikipedia_search(query), crossref_search(query, rows).\n"
                        "Return JSON matching the schema.",
                    ),
                    (
                        "user",
                        "Topic: {topic}\nAudience: {audience}\nConstraints: {constraints}\n"
                        "Return 3-5 research questions and a 2-4 step tool plan.",
                    ),
                ]
            )
            chain = prompt | self._planner.with_structured_output(PlanResult, method="function_calling")
            return chain.invoke({"topic": topic, "audience": audience, "constraints": constraints})

        def synthesize(
            self,
            *,
            topic: str,
            audience: str,
            constraints: str,
            sources: list[dict[str, Any]],
        ) -> str:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Write a concise one-page brief with citations."),
                    (
                        "user",
                        "Topic: {topic}\nAudience: {audience}\nConstraints: {constraints}\n\n"
                        "Sources (JSON):\n{sources}\n\n"
                        "Output markdown with sections: What it is, Why it matters, Risks/Misconceptions, Sources.",
                    ),
                ]
            )
            return (prompt | self._writer).invoke(
                {"topic": topic, "audience": audience, "constraints": constraints, "sources": sources}
            ).content

        def critique(self, *, draft: str, sources: list[dict[str, Any]]) -> CritiqueResult:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Critique the draft briefly. Decide if more research is needed."),
                    (
                        "user",
                        "Draft:\n{draft}\n\nSources (JSON):\n{sources}\n\n"
                        "If citations are missing or draft is too thin, set needs_more_research=true and suggest 1-3 queries.",
                    ),
                ]
            )
            chain = prompt | self._critic.with_structured_output(CritiqueResult, method="function_calling")
            return chain.invoke({"draft": draft, "sources": sources})

    try:
        return OpenAILLM()
    except Exception:
        return RuleBasedLLM()


def get_llm_with_mode(openai_api_key: str | None) -> tuple[LLMAdapter, str]:
    llm = get_llm(openai_api_key)
    if llm.__class__.__name__ == "OpenAILLM":
        return llm, "openai"
    return llm, "rule_based"


def extract_word_cap(constraints: str) -> int | None:
    return _extract_word_cap(constraints)


def truncate_words(markdown: str, max_words: int) -> str:
    return _truncate_words(markdown, max_words)


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _shorten_text(text: str, *, max_sentences: int, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    out = " ".join(sentences[:max_sentences]).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out


def _extract_word_cap(constraints: str) -> int | None:
    m = re.search(r"under\s+(\d+)\s+words?", constraints.lower())
    if not m:
        return None
    try:
        return max(50, int(m.group(1)))
    except Exception:
        return None


def _truncate_words(markdown: str, max_words: int) -> str:
    parts = re.findall(r"\S+|\s+", markdown)
    out: list[str] = []
    wc = 0
    for part in parts:
        if part.isspace():
            out.append(part)
            continue
        wc += 1
        if wc > max_words:
            break
        out.append(part)
    clipped = "".join(out).rstrip()
    if not clipped.endswith("..."):
        clipped += "..."
    return clipped + "\n"
