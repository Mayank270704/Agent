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


def _parse_float(value: str | None, *, default: float) -> float:
    """Parse an environment variable string as a float (Step 33).

    Identical discipline to `_parse_positive_int` just above, for the
    identical reason: unset/blank -> `default` silently (the normal case
    for every deployment that does not override this setting); set but not
    a valid number -> raises `ValueError` at `Settings()` construction
    (process startup), never silently falling back and hiding a deployment
    typo behind behavior that looks like it worked. Range validation (e.g.
    "must be within [-1.0, 1.0]") is deliberately NOT done here, matching
    `_parse_positive_int`'s own reasoning — that is
    `SemanticMemoryRetriever`'s constructor's job, and it only runs when
    semantic memory is enabled.
    """
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"invalid float configuration value {value!r}; expected a number."
        ) from exc


def _parse_optional_float(value: str | None) -> float | None:
    """Parse an optional environment variable string as a float, or `None`
    if unset/blank (Milestone 44.1).

    Unlike `_parse_float` above, there is no numeric default to fall back
    to — "unset" and "explicitly leave Ollama's own default sampling
    behavior alone" are the same state here (see `LLMClient` in
    app/models/llm.py: `None` means no `temperature` is sent in the
    request at all, which is byte-identical to every request before this
    setting existed). Set but not a valid number -> raises `ValueError` at
    `Settings()` construction, matching every other `_parse_*` helper's
    loud-failure discipline — a configuration typo must never be silently
    reinterpreted as "feature stays off."
    """
    if value is None or value.strip() == "":
        return None
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"invalid float configuration value {value!r}; expected a number."
        ) from exc


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional environment variable string as an integer, or
    `None` if unset/blank (Milestone 44.1) — identical reasoning to
    `_parse_optional_float` above, for `OLLAMA_SEED`: `None` means no
    `seed` is sent, byte-identical to every request before this setting
    existed."""
    if value is None or value.strip() == "":
        return None
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"invalid integer configuration value {value!r}; expected a whole number."
        ) from exc


def _parse_cors_origins(value: str | None, *, default: str) -> tuple[str, ...]:
    """Parse CORS_ALLOWED_ORIGINS (Milestone 42): a comma-separated list of
    allowed origins. Unset/blank -> `default` -- the local Next.js dev
    frontend's own default origin (`npm run dev`, port 3000; both the
    `localhost` and `127.0.0.1` forms, since browsers treat them as
    different origins for CORS purposes). This is deliberately NOT a
    wildcard (`allow_origins=["*"]`): Milestone 41 found no frontend
    origin was known yet and correctly deferred CORS entirely; Milestone
    42 is what makes one concrete and known, so the default can finally
    name it explicitly instead of guessing or opening the API to every
    origin. A deployed frontend's real origin is added via this same env
    var, never by widening the default.
    """
    raw = value if value is not None and value.strip() != "" else default
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


_VALID_MEMORY_BACKENDS = {"memory", "sqlite"}


def _parse_memory_backend(value: str | None, *, default: str) -> str:
    """Parse MEMORY_BACKEND (Step 36): unset/blank -> `default`
    ("memory"), silently — the normal case for every deployment that does
    not opt in to persistence. Set but not one of the known backend names
    -> raises `ValueError` immediately, at `Settings()` construction,
    matching every other `_parse_*` helper's rule: a configuration typo
    must never be silently reinterpreted as "feature stays off" or
    "feature uses some other backend" — it must fail loudly instead,
    exactly like `_parse_positive_int`/`_parse_float` already do for a
    malformed number. In particular, this means an unavailable/misspelled
    SQLite backend is never silently downgraded to the in-memory one.
    """
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized not in _VALID_MEMORY_BACKENDS:
        raise ValueError(
            f"invalid MEMORY_BACKEND {value!r}; expected one of {sorted(_VALID_MEMORY_BACKENDS)}."
        )
    return normalized


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
    # Step 33 — SemanticMemoryRetriever's relevance floor. The retriever's
    # OWN class default (app/agent/memory_retriever.py) is -1.0 — a true
    # no-op over cosine similarity's full [-1.0, 1.0] range, deliberately
    # chosen in 16G because the only provider that existed then
    # (DeterministicEmbeddingProvider) produces similarities with no
    # semantic meaning, so no non-zero floor could be honestly calibrated.
    # That reasoning no longer holds once a REAL provider (all-MiniLM-L6-v2,
    # 16H) is in use: an uncalibrated floor at activation time meant
    # "unrelated queries do not retrieve" was NOT actually true of the
    # wiring, only of the retriever class in isolation — a session with
    # fewer than top_k stored memories would surface its least-bad match
    # regardless of how irrelevant it was. 0.3 is not a guess: a direct
    # measurement on this exact model (see tests/test_semantic_memory_
    # evaluation.py) found unrelated-query similarities of 0.013-0.075
    # against a stored fact, and semantically-related-but-differently-
    # worded queries at 0.338-0.645 — 0.3 sits in the middle of that
    # measured gap, matching related queries and rejecting unrelated ones
    # in that evaluation. Like every embedding-dependent constant in this
    # codebase (DEFAULT_DUPLICATE_THRESHOLD, DEFAULT_MIN_CONFIDENCE), this
    # MUST be recalibrated if the embedding model is ever changed.
    semantic_memory_min_similarity: float = field(
        default_factory=lambda: _parse_float(os.getenv("SEMANTIC_MEMORY_MIN_SIMILARITY"), default=0.3)
    )

    # ----------------------------------------------------------------
    # Step 36 — conversation/session persistence backend selection.
    #
    # default_factory for the identical reason as the Step 16I fields
    # above: a plain default would be baked in at module-import time and
    # never observe a later `monkeypatch.setenv(...)` in tests, nor a
    # changed environment in a long-lived process.
    # ----------------------------------------------------------------

    # Default "memory": app/main.py constructs an InMemorySessionMemoryStore,
    # byte-identical to every milestone before 36. "sqlite" selects
    # SQLiteSessionMemoryStore instead (app/persistence/sqlite_memory.py) --
    # no other value is accepted (see _parse_memory_backend above).
    memory_backend: str = field(
        default_factory=lambda: _parse_memory_backend(os.getenv("MEMORY_BACKEND"), default="memory")
    )
    # Only read/used when memory_backend == "sqlite". A plain path string,
    # not validated here -- app/persistence/db.connect() rejects a blank
    # path and creates the parent directory as needed; range/format
    # validation beyond "non-empty" belongs to that call, which only
    # happens when the sqlite backend is actually selected (identical
    # division of responsibility to embedding_device above).
    sqlite_db_path: str = field(
        default_factory=lambda: os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "conversations.db"))
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

    # Milestone 42 — the minimum backend change the frontend milestone
    # needs: which browser origins may call this API cross-origin. Default
    # names the local Next.js dev frontend (frontend/, `npm run dev`, port
    # 3000) explicitly — never a wildcard. default_factory for the same
    # environment-read-timing reason as every field above.
    cors_allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: _parse_cors_origins(
            os.getenv("CORS_ALLOWED_ORIGINS"),
            default="http://localhost:3000,http://127.0.0.1:3000",
        )
    )

    # Milestone 44.1 — optional Ollama generation sampling configuration.
    # Both default to `None` (unset): when unset, `LLMClient` sends the
    # exact same request payload as every milestone before this one (no
    # `options` field at all) — an existing deployment's behavior is
    # byte-identical unless an operator explicitly opts in. This is the
    # SAME shared LLMClient instance used for every LLM call in this
    # codebase (decisions, plan generation, memory extraction — see
    # app/main.py's composition root), so a configured value applies
    # uniformly everywhere, never selectively to just one call site.
    # `OLLAMA_TEMPERATURE` lowers sampling randomness (more literal,
    # less creative output) — plausibly useful for this codebase's
    # JSON-mode decisions and structured extraction, but NOT flipped on
    # by default here: one diagnostic replay is evidence this capability
    # is worth having, not evidence of a specific value to standardize on.
    # `OLLAMA_SEED` pins the random seed, making an identical prompt's
    # output reproducible run-to-run — useful for debugging/reproducing a
    # specific reported failure, not a correctness fix on its own.
    ollama_temperature: float | None = field(
        default_factory=lambda: _parse_optional_float(os.getenv("OLLAMA_TEMPERATURE"))
    )
    ollama_seed: int | None = field(default_factory=lambda: _parse_optional_int(os.getenv("OLLAMA_SEED")))


settings = Settings()
