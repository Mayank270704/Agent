"""Runs one user request through the agent's execution engine.

    ChatService
        |
    AgentOrchestrator
        |                                       (opt-in, Step 11)
        |                                  PlanGenerator -> Plan
        |                                            |
    AgentState (plan attached, or plan=None) <--------+
        |
    AgentLoop  <->  LLMDecisionMaker  <->  ToolRegistry -> Tool -> ToolResult
        |
    AgentResult (read-only summary)

As of Step 7, this is the ONE execution path for handling a user message.
Every request builds a fresh AgentState, hands it to an AgentLoop (bounded,
generic control flow — see app/agent/loop.py) backed by an LLMDecisionMaker
(the LLM-driven "brain" — see app/agent/decision_maker.py), and the loop
runs until the state reaches a terminal status. The final synthesized
answer comes directly from the decision maker's own FINAL decision — there
is no separate answer-synthesis prompt here, because by the time the model
returns FINAL, the tool observations it needed are already in its prompt's
execution history (see LLMDecisionMaker._build_prompt).

Router (app/agent/router.py) is intentionally NOT used here anymore. It
remains in the codebase, independently tested, and is not deleted — see the
Step 7 report for why, and for the documented relationship between Router
and AgentDecision/LLMDecisionMaker. Keeping both wired into this class would
mean two competing execution architectures for the same request, which is
exactly what this step was told to avoid.

Step 11 — plan generation is OPT-IN, not the new default: `plan_generator`
defaults to None, and `process()` only generates and attaches a Plan when
one is explicitly given (Part 2's own wording: a plan "CAN be generated").
This was a deliberate choice, not an oversight — auto-constructing an
LLMPlanGenerator here the way `decision_maker` auto-constructs would (a)
break every existing test that injects a FakeLLM which raises on
`.generate()`, since a real default planner would call it immediately, and
(b) change existing tests' behavior even with a *working* fake planner,
because a generated Plan interacts with AgentLoop's new plan-aware FINAL
check (app/agent/loop.py). ChatService/main.py are untouched, so the live
`/chat` endpoint's behavior is unaffected by this step; planning is fully
built, wired, and tested, but switching it on by default is left to a
future step.

Step 12 — conversation memory, also OPT-IN (`memory: ConversationMemory |
None = None`, default None), for the same reason and using the same
pattern as `plan_generator`: this class's own tests construct orchestrators
that don't care about cross-request history, and a `None` default keeps
them completely unaffected. See app/agent/memory.py for the distinction
between AgentState (one execution) and ConversationMemory (many requests).
When `memory` IS supplied:
- `process()` reads `memory.get_messages()` (previous turns only — the
  CURRENT message stays `AgentState.user_input`, not duplicated into
  `messages`) and seeds the new AgentState's `messages` field with it.
  `LLMDecisionMaker` needed NO changes for this: `_format_history` already
  serializes `state.messages` into its prompt (it just had nothing to show
  before, since nothing ever populated that field).
- Only on a COMPLETED execution does `process()` commit the turn — the
  current user message, then the final answer — back to `memory`. A FAILED
  execution commits nothing (see `_maybe_commit_to_memory`'s docstring for
  why this was a deliberate choice, not an oversight).
- ChatService (Step 12) now owns exactly one `InMemoryConversationMemory`
  and passes it here, replacing its own previously-unused `conversation`
  list — there is now one authoritative source of cross-request history,
  not several independent ones.

Step 14 — episodic memory, also OPT-IN (`episodic_memory: EpisodicMemory |
None = None`, default None), same pattern again. See app/agent/
episodic_memory.py for the ConversationMemory-vs-EpisodicMemory
distinction. When `episodic_memory` IS supplied:
- A `session_id: str | None = None` constructor parameter is REQUIRED
  alongside it (validated eagerly in `__init__`, not deferred to the
  first `process()` call) — every EpisodicMemoryRecord needs a
  `session_id` (Part 2), but `process()`'s signature and AgentState both
  stay exactly as they were (Part 12/26 explicitly forbid adding
  `session_id` to AgentState), so there is nowhere else for it to live.
  This mirrors how `memory` itself is already resolved per-session by the
  caller (ChatService builds a fresh AgentOrchestrator per `ask()` call —
  see app/services/chat.py) and injected at construction time; `session_id`
  simply travels the same way. This constructor requirement — rather than
  silently defaulting to some placeholder session id — is a deliberate
  choice to avoid inventing identity/session semantics that Part 19/23
  explicitly forbid; see the Step 14 report for the full reasoning.
- After a SUCCESSFUL (COMPLETED) execution, `process()` creates exactly
  ONE `EpisodicMemoryRecord` via `_maybe_record_episode` — never per loop
  iteration, never per tool call, never on FAILED (Part 8/9/14). The
  summary is built deterministically from the user's message and the
  final answer only (both bounded/truncated) — no LLM call is made to
  produce it (Part 9/22).
- ChatService is NOT wired to episodic memory in this step (Part 29's
  report items describe only the orchestrator-level foundation) — that
  wiring, and any decision about what session_id the legacy
  `session_id=None` ChatService path would use, is left to a future step.

Step 16E-C — semantic memory retrieval, OPT-IN for the fourth time
(`memory_retriever: MemoryRetriever | None = None`), same pattern and same
reasoning. This is the step that finally connects the 16A-16E-B pipeline to
a live request. When `memory_retriever` IS supplied:
- `session_id` is REQUIRED alongside it, validated eagerly in `__init__`
  for exactly the reason episodic memory requires it: retrieval is
  session-scoped and there is no honest session to search without one.
- `process()` retrieves ONCE, before building AgentState, and attaches the
  resulting MemoryContext to the state (see
  `_retrieve_memory_context_or_none` for why that placement is what makes
  "once per request" structural rather than conventional).
- LLMDecisionMaker then RENDERS `state.memory_context` into its prompt —
  it never retrieves. Embedding and vector search stay entirely on this
  side of the boundary; the prompt layer only formats what it is handed,
  exactly as it does for `state.plan`.
- 16E-C itself was READ-ONLY; the write path arrived in 16E-D below.

Step 16E-D — the semantic memory WRITE path, closing the lifecycle:
`memory_extractor` + `memory_writer` (opt-in, and required together —
neither is useful alone). After a COMPLETED execution, and only then,
`process()` extracts durable facts from the finished interaction exactly
once and writes them to semantic memory, taking provenance from the
episodic event recorded moments earlier in the same call. That makes
episodic_memory a hard requirement for semantic writing (enforced in
__init__): a SemanticMemoryRecord cannot exist without at least one source
event id. A FAILED execution writes nothing, exactly as it records no
episode. See `_maybe_write_semantic_memory`.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.episodic_memory import EpisodicMemory, EpisodicMemoryRecord
from app.agent.loop import AgentLoop, DecisionMaker
from app.agent.memory import ConversationMemory
from app.agent.memory_context import MemoryContext, build_memory_context
from app.agent.memory_extraction import MemoryExtractionError, MemoryExtractor
from app.agent.memory_retriever import MemoryRetriever
from app.agent.memory_writer import MemoryWriter
from app.agent.plan import Plan, PlanGenerator
from app.agent.plan_generator import PlanGenerationError
from app.agent.state import AgentState, AgentStatus, ExecutionError, Observation, ToolCall
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.base import Tool
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 5

# Deterministic, bounded episodic summaries (Part 9/17) — no LLM call, no
# unbounded text. Each half of the summary is truncated independently so a
# very long request never crowds out the answer half (or vice versa).
_EPISODE_SUMMARY_FIELD_LIMIT = 80


def _truncate(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _build_episode_summary(user_message: str, answer: str) -> str:
    request_preview = _truncate(user_message, _EPISODE_SUMMARY_FIELD_LIMIT)
    answer_preview = _truncate(answer, _EPISODE_SUMMARY_FIELD_LIMIT)
    return f"User asked: {request_preview} | Answer: {answer_preview}"


@dataclass(frozen=True)
class AgentResult:
    """A read-only summary of one finished agent execution.

    `answer` is always a usable string regardless of outcome: on COMPLETED
    it is the model's final_answer; on FAILED it is a graceful, apologetic
    message derived from the recorded errors — callers (ChatService today)
    never need to branch on `status` just to get something to show the user.
    """

    answer: str
    status: AgentStatus
    steps: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    errors: list[ExecutionError] = field(default_factory=list)


class AgentOrchestrator:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        decision_maker: DecisionMaker | None = None,
        plan_generator: PlanGenerator | None = None,
        memory: ConversationMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        memory_retriever: MemoryRetriever | None = None,
        memory_extractor: MemoryExtractor | None = None,
        memory_writer: MemoryWriter | None = None,
        session_id: str | None = None,
        web_search_tool: Tool | None = None,
        time_tool: Tool | None = None,
        date_tool: Tool | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.llm = llm_client or LLMClient(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.ollama_base_url,
        )

        if tool_registry is not None:
            self.tools = tool_registry
        else:
            self.tools = ToolRegistry()
            self.tools.register(web_search_tool or WebSearchTool())
            self.tools.register(time_tool or TimeTool())
            self.tools.register(date_tool or DateTool())

        self.decision_maker = decision_maker or LLMDecisionMaker(llm_client=self.llm, tool_registry=self.tools)
        # Deliberately NOT defaulted like decision_maker is — see the module
        # docstring. None means "no plan generation," exactly like before
        # Step 11; a caller opts in by passing one explicitly.
        self.plan_generator = plan_generator
        # Same pattern, same reasoning: None means "no cross-request
        # conversation history," exactly like before Step 12.
        self.memory = memory
        # Same pattern again: None means "no episodic recording," exactly
        # like before Step 14. See the module docstring for why session_id
        # is required (and validated eagerly, here) whenever episodic_memory
        # is supplied.
        self.episodic_memory = episodic_memory
        # Same opt-in pattern again (Step 16E-C): None means "no semantic
        # memory retrieval," exactly like before. Retrieval is READ-ONLY —
        # nothing here ever writes a semantic memory.
        self.memory_retriever = memory_retriever
        self.session_id = session_id.strip() if isinstance(session_id, str) else session_id
        if self.episodic_memory is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise ValueError("session_id is required (and must be non-empty) when episodic_memory is supplied.")
        if self.memory_retriever is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise ValueError("session_id is required (and must be non-empty) when memory_retriever is supplied.")

        # Step 16E-D — the WRITE path. Both halves are required together:
        # an extractor with nowhere to write proposes facts that vanish,
        # and a writer with nothing to extract can never be called.
        self.memory_extractor = memory_extractor
        self.memory_writer = memory_writer
        if (self.memory_extractor is None) != (self.memory_writer is None):
            raise ValueError("memory_extractor and memory_writer must be supplied together.")
        if self.memory_writer is not None:
            if not isinstance(self.session_id, str) or not self.session_id.strip():
                raise ValueError(
                    "session_id is required (and must be non-empty) when memory_extractor/memory_writer are supplied."
                )
            if self.episodic_memory is None:
                # Every SemanticMemoryRecord requires at least one source
                # event id, and the episodic record created for this same
                # interaction IS that source. Without episodic memory there
                # is no provenance to attach, so the write could not
                # produce a valid record at all.
                raise ValueError(
                    "episodic_memory is required when memory_extractor/memory_writer are supplied, "
                    "because semantic memories take their provenance from the episodic event."
                )
        self.loop = AgentLoop(
            decision_maker=self.decision_maker,
            tool_registry=self.tools,
            max_iterations=max_iterations,
        )

    def process(self, user_message: str) -> AgentResult:
        if user_message is None or not str(user_message).strip():
            raise ValueError("User message cannot be empty.")

        cleaned_message = user_message.strip()
        plan = self._generate_plan_or_none(cleaned_message)
        previous_messages = self.memory.get_messages() if self.memory is not None else []
        memory_context = self._retrieve_memory_context_or_none(cleaned_message)
        state = AgentState(
            user_input=cleaned_message,
            messages=previous_messages,
            plan=plan,
            memory_context=memory_context,
        )
        self.loop.run(state)

        answer = state.final_answer if state.status == AgentStatus.COMPLETED else self._failure_answer(state)
        self._maybe_commit_to_memory(cleaned_message, answer, state)
        episode = self._maybe_record_episode(cleaned_message, answer, state)
        self._maybe_write_semantic_memory(cleaned_message, answer, state, episode)

        return AgentResult(
            answer=answer,
            status=state.status,
            steps=state.step,
            tool_calls=list(state.tool_calls),
            observations=list(state.observations),
            errors=list(state.errors),
        )

    def _failure_answer(self, state: AgentState) -> str:
        if state.errors:
            return f"I could not complete this request: {state.errors[-1].message}"
        return "I could not complete this request due to an unexpected failure."

    def _generate_plan_or_none(self, user_message: str) -> Plan | None:
        """No-op (returns None) unless a plan_generator was explicitly
        injected — see the module docstring for why this stays opt-in.

        A PlanGenerationError (malformed/invalid model output — see
        app/agent/plan_generator.py) is caught and degrades gracefully to
        "no plan," exactly the already-supported plan=None case, rather
        than failing the whole request over a planning-layer hiccup. A
        RuntimeError (e.g. Ollama unreachable) is NOT caught here — it
        propagates the same way every other LLM-call failure in this
        codebase does, since converting it into "no plan" would only defer
        the identical failure to the very next LLM call inside the loop.
        """
        if self.plan_generator is None:
            return None

        try:
            return self.plan_generator.generate(user_message)
        except PlanGenerationError as exc:
            logger.warning("Plan generation failed for %r: %s — continuing without a plan.", user_message, exc)
            return None

    def _retrieve_memory_context_or_none(self, user_message: str) -> MemoryContext | None:
        """No-op (returns None) unless a memory_retriever was explicitly
        injected — see the module docstring for why this stays opt-in.

        Called EXACTLY ONCE per `process()` call, before AgentState is
        built and therefore before AgentLoop runs. That placement is what
        guarantees the "retrieve once per user request" requirement
        structurally rather than by convention: the loop, the decision
        maker, and the tools never hold a reference to the retriever at
        all, so no number of loop iterations can trigger another
        embedding call or vector search. The retrieved context is then a
        fixed, immutable snapshot for the whole execution.

        Session isolation is delegated entirely to the retriever
        (app/agent/memory_retriever.py), which searches only the named
        session's own vector partition and re-verifies every resolved
        record's ownership. This method adds no filtering of its own and
        deliberately does not bypass that abstraction — it simply passes
        `self.session_id`, which is non-empty whenever a retriever is
        present (enforced in __init__).

        Exceptions are deliberately NOT caught here. The only failures the
        retriever raises today are input-validation errors and session-
        isolation violations, and a session-isolation violation is a
        security signal that must never be silently degraded into "no
        memory." Making retrieval failure non-fatal needs a dedicated
        error type to distinguish infrastructure failure from an
        integrity violation — deferred, see the Step 16E-C report.
        """
        if self.memory_retriever is None:
            return None

        retrieved = self.memory_retriever.retrieve(self.session_id, user_message)
        return build_memory_context(self.session_id, retrieved)

    def _maybe_commit_to_memory(self, user_message: str, answer: str, state: AgentState) -> None:
        """No-op unless memory was explicitly injected. Otherwise, commits
        the turn (user message, then the final answer) ONLY when the
        execution actually COMPLETED.

        Deliberate choice (Part 6): a FAILED execution's answer is a
        graceful apology, not a real assistant reply — persisting it would
        bake a broken turn into conversation history and (worse) make the
        NEXT request's prompt cite "assistant previously said: I could not
        complete this request..." as if that were meaningful prior context.
        The user's message itself is also not committed on failure: they
        are free to resend it, and half of a turn (their message with no
        matching reply) is not a coherent conversation entry either.
        """
        if self.memory is None or state.status != AgentStatus.COMPLETED:
            return
        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(answer)

    def _maybe_record_episode(
        self, user_message: str, answer: str, state: AgentState
    ) -> EpisodicMemoryRecord | None:
        """No-op unless episodic_memory was explicitly injected. Otherwise,
        records exactly ONE EpisodicMemoryRecord per successfully COMPLETED
        execution (Part 8/9/14) — never for a FAILED execution, never once
        per loop iteration or tool call.

        The summary is deterministic (Part 9/22): built only from the
        user's message and the final answer, each independently bounded —
        no LLM call, no chain-of-thought, no raw tool payloads. `metadata`
        stays small and structured (Part 17): just the step count and how
        many tools were used, nothing that could leak internal execution
        detail.

        Returns the record it created (Step 16E-D) so the caller can use
        its `event_id` as provenance for any semantic memory extracted
        from the same interaction, or None when no episode was recorded.
        """
        if self.episodic_memory is None or state.status != AgentStatus.COMPLETED:
            return None
        record = EpisodicMemoryRecord(
            event_id=str(uuid.uuid4()),
            session_id=self.session_id,
            event_type="conversation_completed",
            summary=_build_episode_summary(user_message, answer),
            timestamp=datetime.now(timezone.utc),
            metadata={"steps": state.step, "tool_count": len(state.tool_calls)},
        )
        self.episodic_memory.add(record)
        return record

    def _maybe_write_semantic_memory(
        self,
        user_message: str,
        answer: str,
        state: AgentState,
        episode: EpisodicMemoryRecord | None,
    ) -> None:
        """No-op unless BOTH a memory_extractor and a memory_writer were
        injected. Otherwise extracts durable facts from the completed
        interaction exactly ONCE and writes them to semantic memory.

        Lifecycle, matching episodic memory's rule exactly: this runs only
        after the loop has finished and only when the execution actually
        COMPLETED. A FAILED execution writes nothing — its "answer" is a
        graceful apology, not a real reply, and distilling durable facts
        about the user from a failed turn would bake noise into long-term
        memory. `episode is None` is an equivalent guard: no episode means
        no provenance, and an unsourced semantic memory cannot be built.

        Timing: once per `process()` call, after `loop.run()` has returned.
        The loop, decision maker and tools never hold a reference to the
        extractor or writer, so no number of iterations and no tool
        observation can trigger an extraction — the same structural
        guarantee `_retrieve_memory_context_or_none` gives on the read
        side.

        Source material is the completed interaction only: the user's
        message and the final answer. Intermediate tool observations are
        deliberately NOT passed — they are transient external content
        (search results, timestamps), not durable facts about the user,
        and feeding them to an extractor would invite exactly the
        transcript-shaped memories this pipeline is meant to avoid.

        A MemoryExtractionError degrades gracefully to "nothing extracted"
        rather than failing the request: the user's answer has already
        been produced, and an unparseable extraction is a quality problem,
        not a reason to discard a successful turn. This mirrors
        `_generate_plan_or_none`'s handling of PlanGenerationError. The
        catch is deliberately narrow — it cannot swallow the writer's
        validation or session errors, which stay loud.
        """
        if self.memory_extractor is None or self.memory_writer is None:
            return
        if state.status != AgentStatus.COMPLETED or episode is None:
            return

        try:
            candidates = self.memory_extractor.extract(user_message, answer)
        except MemoryExtractionError as exc:
            logger.warning("Memory extraction failed for %r: %s — storing no semantic memory.", user_message, exc)
            return

        if not candidates:
            return

        self.memory_writer.write(
            session_id=self.session_id,
            candidates=candidates,
            source_event_ids=(episode.event_id,),
        )
