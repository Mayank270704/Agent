"""Decision-making components for the chatbot."""

from app.agent.decision_maker import DecisionParseError, LLMDecisionMaker
from app.agent.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider, Vector
from app.agent.episodic_memory import EpisodicMemory, EpisodicMemoryRecord, InMemoryEpisodicMemory
from app.agent.evaluation import (
    CaseResult,
    CaseVerdict,
    ComparisonSummary,
    EvaluationCase,
    EvaluationSummary,
    NoHintRouter,
    ROUTING_EVALUATION_CASES,
    compare,
    evaluate_all,
    evaluate_case,
    format_comparison_report,
    format_report,
    summarize,
)
from app.agent.local_embeddings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    EmbeddingComputeError,
    EmbeddingError,
    EmbeddingModelLoadError,
    LocalEmbeddingProvider,
)
from app.agent.loop import ActionType, AgentDecision, AgentLoop, DecisionMaker, DecisionMakerError
from app.agent.memory_context import MemoryContext, MemoryContextItem, build_memory_context
from app.agent.memory_extraction import (
    LLMMemoryExtractor,
    MemoryCandidate,
    MemoryExtractionError,
    MemoryExtractor,
)
from app.agent.memory_formatting import MEMORY_CONTEXT_LABEL, format_memory_context
from app.agent.memory_retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_TOP_K,
    DEFAULT_TOP_K,
    MemoryRetriever,
    RetrievedMemory,
    SemanticMemoryRetriever,
)
from app.agent.memory_writer import MemoryWriter, SemanticMemoryWriter, rejection_reason
from app.agent.memory import (
    ConversationMemory,
    InMemoryConversationMemory,
    InMemorySessionMemoryStore,
    SessionMemoryStore,
)
from app.agent.orchestrator import AgentOrchestrator, AgentResult
from app.agent.plan import Plan, PlanGenerator, PlanStatus, PlanStep
from app.agent.plan_generator import LLMPlanGenerator, PlanGenerationError
from app.agent.router import Router, RouterDecision, RoutingHint
from app.agent.semantic_memory import (
    InMemorySemanticMemory,
    MemorySessionIsolationError,
    SemanticMemoryRecord,
    SemanticMemoryStore,
)
from app.agent.vector_index import InMemoryVectorIndex, VectorIndex, VectorSearchResult, cosine_similarity
from app.agent.state import AgentState, AgentStatus, CorrectionNote, ExecutionError, Observation, ToolCall
from app.agent.reliability import (
    DEFAULT_MAX_CORRECTIONS,
    BudgetedCorrectionPolicy,
    CorrectionAction,
    CorrectionPolicy,
    CorrectionVerdict,
    Failure,
    FailureCategory,
)
from app.agent.permissions import (
    AllowlistPermissionPolicy,
    ExecutionContext,
    PermissionDecision,
    PermissionPolicy,
)
from app.agent.tool_execution import ConfirmationRequiredError, PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolNotFoundError, ToolRegistrationError, ToolRegistry
from app.tools.base import RiskLevel, ToolCapability, ToolDescriptor

__all__ = [
    "Router",
    "RouterDecision",
    "RoutingHint",
    "ToolRegistry",
    "ToolRegistrationError",
    "ToolNotFoundError",
    "FailureCategory",
    "Failure",
    "CorrectionAction",
    "CorrectionVerdict",
    "CorrectionPolicy",
    "BudgetedCorrectionPolicy",
    "DEFAULT_MAX_CORRECTIONS",
    "ToolDescriptor",
    "ToolCapability",
    "RiskLevel",
    "PermissionDecision",
    "PermissionPolicy",
    "AllowlistPermissionPolicy",
    "ExecutionContext",
    "ToolExecutionGate",
    "PermissionDeniedError",
    "ConfirmationRequiredError",
    "AgentState",
    "AgentStatus",
    "ToolCall",
    "Observation",
    "ExecutionError",
    "CorrectionNote",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanGenerator",
    "LLMPlanGenerator",
    "PlanGenerationError",
    "AgentLoop",
    "AgentDecision",
    "ActionType",
    "DecisionMaker",
    "DecisionMakerError",
    "LLMDecisionMaker",
    "DecisionParseError",
    "AgentOrchestrator",
    "AgentResult",
    "EvaluationCase",
    "CaseVerdict",
    "CaseResult",
    "EvaluationSummary",
    "evaluate_case",
    "evaluate_all",
    "summarize",
    "format_report",
    "ROUTING_EVALUATION_CASES",
    "NoHintRouter",
    "ComparisonSummary",
    "compare",
    "format_comparison_report",
    "ConversationMemory",
    "InMemoryConversationMemory",
    "SessionMemoryStore",
    "InMemorySessionMemoryStore",
    "EpisodicMemory",
    "EpisodicMemoryRecord",
    "InMemoryEpisodicMemory",
    "SemanticMemoryStore",
    "SemanticMemoryRecord",
    "InMemorySemanticMemory",
    "MemorySessionIsolationError",
    "EmbeddingProvider",
    "DeterministicEmbeddingProvider",
    "Vector",
    "LocalEmbeddingProvider",
    "EmbeddingError",
    "EmbeddingModelLoadError",
    "EmbeddingComputeError",
    "DEFAULT_EMBEDDING_MODEL_NAME",
    "VectorIndex",
    "InMemoryVectorIndex",
    "VectorSearchResult",
    "cosine_similarity",
    "MemoryRetriever",
    "RetrievedMemory",
    "SemanticMemoryRetriever",
    "DEFAULT_TOP_K",
    "DEFAULT_MAX_TOP_K",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "MemoryContext",
    "MemoryContextItem",
    "build_memory_context",
    "format_memory_context",
    "MEMORY_CONTEXT_LABEL",
    "MemoryCandidate",
    "MemoryExtractor",
    "LLMMemoryExtractor",
    "MemoryExtractionError",
    "MemoryWriter",
    "SemanticMemoryWriter",
    "rejection_reason",
]
