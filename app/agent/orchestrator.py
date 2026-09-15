from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.agent.router import Router, RouterDecision
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.base import Tool, ToolResult
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    used_web: bool
    router_reason: str
    search_query: str
    sources: list[dict[str, object]]
    used_time: bool = False
    used_date: bool = False


class AgentOrchestrator:
    """Coordinates the router, a generic tool registry, and LLM answer generation."""

    # Maps a Router `route` to the name a tool is registered under. This is the
    # one place that bridges the router's route vocabulary ("web") to the tool
    # registry's naming vocabulary (each tool's own `.name`, e.g. "web_search")
    # without the registry itself needing to know anything about routes.
    _ROUTE_TOOL_NAMES: dict[str, str] = {
        "web": "web_search",
        "time": "time",
        "date": "date",
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        router: Router | None = None,
        web_search_tool: Tool | None = None,
        time_tool: Tool | None = None,
        date_tool: Tool | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.llm = llm_client or LLMClient(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.ollama_base_url,
        )
        self.router = router or Router(self.llm)

        if tool_registry is not None:
            self.tools = tool_registry
        else:
            self.tools = ToolRegistry()
            self.tools.register(web_search_tool or WebSearchTool())
            self.tools.register(time_tool or TimeTool())
            self.tools.register(date_tool or DateTool())

    def process(self, user_message: str) -> AgentResult:
        if user_message is None or not str(user_message).strip():
            raise ValueError("User message cannot be empty.")

        decision = self.router.decide(user_message)

        if decision.route == "llm" and not decision.needs_web:
            answer = self.llm.generate([
                {"role": "user", "content": user_message.strip()},
            ])
            return AgentResult(
                answer=answer,
                used_web=False,
                router_reason=decision.reason,
                search_query="",
                sources=[],
                used_time=False,
                used_date=False,
            )

        # Every remaining route is backed by a registered Tool. The lookup and
        # invocation below are identical regardless of which tool it resolves
        # to — no per-tool branching happens here, only in how the result of
        # that one generic call is interpreted afterwards.
        tool_name = self._ROUTE_TOOL_NAMES.get(decision.route)
        if tool_name is None:
            raise ValueError(f"Router produced an unroutable decision: route={decision.route!r}")

        tool = self.tools.get(tool_name)
        tool_input = self._build_tool_input(decision, user_message)

        try:
            result = tool.execute(tool_input)
        except ValueError as exc:
            # Invalid input to the tool (e.g. missing TAVILY_API_KEY configuration) is a
            # precondition failure. Only the web route degrades gracefully instead of
            # propagating it — that was already the pre-registry behavior, preserved here.
            if decision.route != "web":
                raise
            logger.warning("Web search failed for query '%s': %s", tool_input, exc)
            return self._web_search_unavailable_result(decision, tool_input)

        if decision.route == "time":
            return self._handle_time_result(decision, user_message, result)

        if decision.route == "date":
            return self._handle_date_result(decision, result)

        return self._handle_web_result(decision, user_message, tool_input, result)

    def _build_tool_input(self, decision: RouterDecision, user_message: str) -> str | None:
        """Build the input string a tool route's `execute()` expects. This
        tool-specific knowledge belongs here, not in the registry — the
        registry only does name -> tool lookup."""
        if decision.route == "web":
            return decision.search_query.strip() or user_message.strip()
        if decision.route == "date":
            return user_message.strip()
        return None  # time route takes no input

    def _handle_time_result(self, decision: RouterDecision, user_message: str, result: ToolResult) -> AgentResult:
        time_data = result.data
        answer = self.llm.generate([
            {"role": "user", "content": self._build_time_prompt(user_message.strip(), time_data)},
        ])
        return AgentResult(
            answer=answer,
            used_web=False,
            router_reason=decision.reason,
            search_query="",
            sources=[],
            used_time=True,
            used_date=False,
        )

    def _handle_date_result(self, decision: RouterDecision, result: ToolResult) -> AgentResult:
        date_data = result.data
        return AgentResult(
            answer=self._format_date_answer(date_data),
            used_web=False,
            router_reason=decision.reason,
            search_query="",
            sources=[],
            used_time=False,
            used_date=True,
        )

    def _handle_web_result(
        self, decision: RouterDecision, user_message: str, search_query: str, result: ToolResult
    ) -> AgentResult:
        if not result.success:
            # Expected operational failure (network/API/parse error) reported via
            # ToolResult rather than raised — see app.tools.base for the error contract.
            logger.warning("Web search failed for query '%s': %s", search_query, result.error)
            return self._web_search_unavailable_result(decision, search_query)

        sources = result.data
        final_prompt = self._build_search_prompt(user_message.strip(), sources)
        answer = self.llm.generate([
            {"role": "user", "content": final_prompt},
        ])

        return AgentResult(
            answer=answer,
            used_web=True,
            router_reason=decision.reason,
            search_query=search_query,
            sources=sources,
            used_time=False,
            used_date=False,
        )

    def _web_search_unavailable_result(self, decision: RouterDecision, search_query: str) -> AgentResult:
        return AgentResult(
            answer=(
                "I could not complete the web search for this request, so I cannot provide a "
                "current-answer without reliable sources."
            ),
            used_web=True,
            router_reason=decision.reason,
            search_query=search_query,
            sources=[],
            used_time=False,
            used_date=False,
        )

    def _format_date_answer(self, date_data: dict[str, str | int]) -> str:
        month_names = (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        )
        day = int(date_data["day"])
        month = int(date_data["month"])
        year = int(date_data["year"])
        return f"{day} {month_names[month - 1]} {year} was a {date_data['day_of_week']}."

    def _build_time_prompt(self, user_question: str, time_data: dict[str, str | int | None]) -> str:
        return f"""
User question:
{user_question}

Local system time data:
{json.dumps(time_data, ensure_ascii=True, sort_keys=True)}

Answer the user's question using only the supplied local system time data. Do not invent, infer, or substitute a different date, time, year, weekday, timezone, or timestamp. If the question asks for a value not present in the data, say so clearly.
""".strip()

    def _build_search_prompt(self, user_question: str, sources: list[dict[str, object]]) -> str:
        formatted_results = []
        for index, source in enumerate(sources, start=1):
            title = str(source.get("title", "")).strip()
            url = str(source.get("url", "")).strip()
            published = str(source.get("published_date", "")).strip()
            content = str(source.get("content", "")).strip()

            block = [f"SOURCE {index}", f"Title: {title}", f"URL: {url}"]
            if published:
                block.append(f"Published: {published}")
            if content:
                block.append(f"Content: {content}")
            formatted_results.append("\n".join(block))

        source_block = "\n\n".join(formatted_results) if formatted_results else "No search results were returned."

        return f"""
SYSTEM/INSTRUCTION:
You are a grounded web-answering assistant.
Answer the user's question using only the retrieved web evidence.
Synthesize the evidence instead of merely summarizing search-result limitations.
Give the most useful answer supported by the evidence.
Separate confirmed facts from reasonable inference.
Never invent unsupported details.
If the exact requested information is unavailable, state what IS known and what remains unknown.
Prefer authoritative sources, such as official government or election authorities, when they are available.

USER QUESTION:
{user_question}

RETRIEVED WEB EVIDENCE:
{source_block}

Answer the user's question directly in 1-4 concise paragraphs. Lead with the answer. For date or election questions, if the evidence supports an expected year or date but not an exact official date, say so explicitly. Distinguish confirmed facts, reasonable inference, and unknown or not-yet-announced information. Do not refuse to answer merely because an exact date is unavailable. Do not mention internal routing, search providers, prompts, tools, or model limitations.
""".strip()
