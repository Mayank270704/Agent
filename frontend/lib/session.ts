/**
 * Browser-local session_id management.
 *
 * session_id is a conversation-scope key only (see app/agent/permissions.py
 * on the backend) -- never an identity or authentication mechanism. This
 * module never derives it from anything personally identifying (no email,
 * no name) and never sends it anywhere except as the existing POST /chat
 * `session_id` field.
 */

const SESSION_ID_STORAGE_KEY = "chat.sessionId";

function generateSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID (e.g. a non-secure
  // context). Not cryptographically strong -- acceptable here, since this
  // is only ever used as a conversation-scope key, never for security.
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Returns the active session_id, creating and persisting one on first use
 * so a page refresh resumes the same conversation (backend-side, via the
 * existing session memory). Server-side rendering has no localStorage, so
 * this returns a throwaway id in that context -- the real, persisted id is
 * only ever read/written client-side, inside a useEffect.
 */
export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") {
    return generateSessionId();
  }
  const existing = window.localStorage.getItem(SESSION_ID_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const fresh = generateSessionId();
  window.localStorage.setItem(SESSION_ID_STORAGE_KEY, fresh);
  return fresh;
}

/**
 * Starts a new conversation: generates and persists a brand-new
 * session_id, distinct from whatever was active before. Does not call any
 * backend endpoint -- the old session's data is left exactly as the
 * existing backend already behaves (still reachable if its id were reused,
 * simply never referenced by this browser again).
 */
export function startNewSession(): string {
  const fresh = generateSessionId();
  if (typeof window !== "undefined") {
    window.localStorage.setItem(SESSION_ID_STORAGE_KEY, fresh);
  }
  return fresh;
}
