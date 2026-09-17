from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    """Parse an environment variable string as a boolean (Step 16I).

    Deliberately narrow: only a small set of common truthy spellings
    ("true", "1", "yes", "on", case-insensitive after stripping
    whitespace) count as True. Every other value — including an empty
    string, "0", "no", "off", and any unrecognized text such as "maybe" or
    a typo — is treated as False. There is no "falsy-looking but actually
    truthy" spelling: the failure mode of an unrecognized or malformed
    value is always "feature stays off," never "feature turns on
    unexpectedly." A missing variable (`value is None`) uses `default`
    rather than being treated as an explicit "false" — the two are
    different questions ("not set" vs. "set to a falsy string") even
    though both currently resolve to the same default.
    """
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _parse_positive_int(value: str | None, *, default: int) -> int:
    """Parse an environment variable string as an integer (Step 16I).

    Unset or blank -> `default`, silently: that is the normal, expected
    case for every deployment that does not override this setting. Set
    but not a valid integer -> raises `ValueError` immediately, at
    `Settings()` construction (i.e. at process startup). A malformed
    configuration value is a deployment bug, and silently falling back to
    the default would hide that bug behind behavior that looks like it
    worked — the same "do not hide configuration bugs" rule the semantic
    memory retrieval/write path already follows (16G/16I). Range
    validation (e.g. "must be >= 1") is deliberately NOT done here: that
    is the SemanticMemoryWriter constructor's own job (16F), and it only
    actually runs when semantic memory is enabled — duplicating it here
    would let an unrelated-when-disabled value crash a deployment that
    never even turns the feature on.
    """
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"invalid integer configuration value {value!r}; expected a whole number."
        ) from exc


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Chatbot API")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    # Step 16H: which local sentence-embedding model LocalEmbeddingProvider
    # loads. Env plumbing only — no API key. The literal MUST stay equal to
    # local_embeddings.DEFAULT_EMBEDDING_MODEL_NAME; a test asserts it, so
    # the two defaults cannot drift apart. Kept as a plain default (read
    # once, at module import) rather than default_factory: nothing in this
    # codebase needs to observe a changed EMBEDDING_MODEL_NAME within one
    # process's lifetime, unlike semantic_memory_enabled below.
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ----------------------------------------------------------------
    # Step 16I — semantic memory activation.
    #
    # These three fields use `default_factory`, NOT a plain
    # `= os.getenv(...)` default like every field above. A dataclass
    # field's plain default expression is evaluated exactly ONCE, when
    # the class BODY executes (i.e. when this module is first imported) —
    # verified directly: re-running `os.environ[...] = ...; Settings()`
    # after import keeps returning the value that was baked in at import
    # time, never the new one. That makes a plain default both untestable
    # (`monkeypatch.setenv(...)` before constructing a fresh `Settings()`
    # would have no effect) and silently stale in any long-lived process
    # that reads a changed environment. `default_factory` is a callable
    # invoked at EACH `Settings()` construction, so it reads the current
    # environment every time — which is what "read at the correct time"
    # means here, and is exactly the property the 16I design review
    # required before this flag could gate real activation.
    # ----------------------------------------------------------------

    # Default OFF. Disabled means app/main.py constructs no embedding
    # provider, no vector index, no semantic store — imports no ML
    # library at all — and passes None for all three ChatService memory
    # collaborators, byte-identical to every milestone before 16I.
    semantic_memory_enabled: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("SEMANTIC_MEMORY_ENABLED"), default=False)
    )
    # CPU vs GPU device for LocalEmbeddingProvider. No format validation
    # here (any non-empty string is syntactically acceptable); the
    # provider itself rejects blank/invalid values at construction, and
    # that construction only happens when semantic memory is enabled.
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    # SemanticMemoryWriter's retention cap — the ONLY retention mechanism
    # semantic memory uses (16F/16I); InMemorySemanticMemory itself is
    # always constructed uncapped so store and index eviction can never
    # drift apart. See app/semantic_memory_wiring.py.
    semantic_memory_max_records_per_session: int = field(
        default_factory=lambda: _parse_positive_int(
            os.getenv("SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION"), default=200
        )
    )

    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # Milestone 19. Default OFF, identical idiom to semantic_memory_enabled
    # above. Disabled means app/main.py constructs no EventSink at all and
    # passes `event_sink=None` to ChatService — no AgentEvent, no
    # EventEmitter, and no telemetry-related log line is ever produced,
    # byte-identical to every milestone before 19.
    telemetry_enabled: bool = field(default_factory=lambda: _parse_bool(os.getenv("TELEMETRY_ENABLED"), default=False))

    # Milestone 23: bounds the TOTAL wall-clock lifetime of one /chat
    # request (all decide/tool-execution iterations combined) — separate
    # from, and not derived from, LLMClient's own per-call timeout (60s,
    # app/models/llm.py). See app/agent/loop.py's `deadline` handling for
    # why this is a cooperative check between iterations, not a hard
    # cancellation. default_factory (not a plain default) for the same
    # environment-read-timing reason as the semantic-memory fields above.
    request_timeout_seconds: int = field(
        default_factory=lambda: _parse_positive_int(os.getenv("REQUEST_TIMEOUT_SECONDS"), default=120)
    )


settings = Settings()
