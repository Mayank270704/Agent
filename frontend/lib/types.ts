/**
 * Shared types for the chat frontend. Deliberately small and mirrors
 * exactly what the backend API (app/main.py) actually sends/accepts --
 * nothing here represents a capability the backend doesn't have.
 */

export type MessageRole = "user" | "assistant";

export type Message = {
  role: MessageRole;
  content: string;
};

/** POST /chat request body (app/main.py's ChatRequest). */
export type ChatRequest = {
  message: string;
  session_id?: string;
};

/** POST /chat response body (app/main.py's ChatResponse). */
export type ChatResponse = {
  reply: string;
};

/**
 * GET /health response body (app/main.py). Kept loose/partial on purpose:
 * this UI only ever reads `status`, never the more detailed persistence
 * fields -- showing those to an end user would expose internal backend
 * state (schema version, backend type) nobody using the chat needs to see.
 */
export type HealthResponse = {
  status: string;
};

/**
 * A safe, user-facing API failure. `message` is always text that is safe
 * to render directly -- either the backend's own already-sanitized error
 * detail (app/main.py never leaks secrets/paths/stack traces in `detail`,
 * audited in Milestone 41) or a generic client-authored fallback. Never
 * constructed from a raw exception/stack trace.
 */
export class ApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
