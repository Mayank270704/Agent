/**
 * The ONE place that talks to the backend API (app/main.py). No UI
 * component calls fetch() directly -- every request goes through here, so
 * error handling, the base URL, and response parsing stay in one place.
 */
import { ApiError, type ChatRequest, type ChatResponse, type HealthResponse } from "./types";

/**
 * Configurable via NEXT_PUBLIC_API_BASE_URL (a public, non-secret value --
 * see frontend/.env.local.example). Defaults to the documented local
 * backend address so `npm run dev` works out of the box against
 * `python -m app.main`.
 */
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/+$/, "");

// Slightly above the backend's own default REQUEST_TIMEOUT_SECONDS=120
// (app/config.py) -- the backend's cooperative deadline already guarantees
// POST /chat responds well within this window under normal operation; this
// is a client-side safety net for a genuinely stuck connection, not a
// duplicate of the backend's own timeout logic.
const REQUEST_TIMEOUT_MS = 130_000;

/**
 * Reads a backend error response's `detail` field, handling both shapes
 * app/main.py can send: a plain string (its own HTTPException branches)
 * and a FastAPI/Pydantic validation-error array (422 responses). Never
 * surfaces anything beyond what the backend itself already decided was
 * safe to return -- this function adds no new information, it only
 * extracts a displayable string from it.
 */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail.trim()) {
        return detail;
      }
      if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0] as { msg?: unknown };
        if (first && typeof first.msg === "string" && first.msg.trim()) {
          return first.msg;
        }
      }
    }
  } catch {
    // Response body wasn't JSON (or was empty) -- fall through below.
  }
  return `The agent could not process that request (HTTP ${response.status}).`;
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") {
      throw new ApiError("The request took too long and was cancelled. Please try again.");
    }
    throw new ApiError("Unable to reach the agent. Check that the backend is running.");
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("The agent returned an unexpected response.", response.status);
  }

  if (!data || typeof data !== "object" || typeof (data as Partial<ChatResponse>).reply !== "string") {
    throw new ApiError("The agent returned an unexpected response.", response.status);
  }

  return data as ChatResponse;
}

/**
 * Returns the parsed GET /health body, or `null` if the backend could not
 * be reached or returned something unparseable. A 503 (degraded
 * persistence) is still a successful, parseable response from this
 * function's point of view -- only a network failure or malformed body
 * is treated as "unknown".
 */
export async function getHealth(): Promise<HealthResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data: unknown = await response.json();
    if (data && typeof data === "object" && typeof (data as Partial<HealthResponse>).status === "string") {
      return data as HealthResponse;
    }
    return null;
  } catch {
    return null;
  }
}
