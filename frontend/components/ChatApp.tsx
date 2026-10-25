"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, sendChatMessage } from "@/lib/api";
import { getOrCreateSessionId, startNewSession } from "@/lib/session";
import { ApiError, type Message } from "@/lib/types";
import ChatInput from "./ChatInput";
import MessageBubble from "./MessageBubble";
import Sidebar from "./Sidebar";
import StatusIndicator, { type ConnectionStatus } from "./StatusIndicator";
import TypingIndicator from "./TypingIndicator";

// Per-session UI cache only (sessionStorage, not localStorage): a
// convenience so a page REFRESH keeps the visible transcript, while a new
// browser tab/session still starts from what the backend actually has.
// The backend (ConversationMemory, optionally SQLite-persisted) remains
// the single source of truth -- this cache is never read by anything
// except this component, and is discarded whenever "New Chat" starts a
// fresh session_id.
function cacheKeyFor(sessionId: string): string {
  return `chat.messages.${sessionId}`;
}

function loadCachedMessages(sessionId: string): Message[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(cacheKeyFor(sessionId));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is Message =>
        !!item &&
        typeof item === "object" &&
        (item.role === "user" || item.role === "assistant") &&
        typeof item.content === "string",
    );
  } catch {
    return [];
  }
}

export default function ChatApp() {
  const [sessionId, setSessionId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("checking");
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  // Session + cached transcript are resolved client-side only, once, on
  // mount -- localStorage/sessionStorage do not exist during server
  // rendering, and a session_id must never be baked into server output
  // (it would otherwise be shared by every visitor's first paint). This is
  // the canonical "synchronize with an external system" effect (reading
  // browser storage) React's own docs describe; the two setState calls
  // below are batched into a single render by React 18+, not a cascade.
  useEffect(() => {
    const id = getOrCreateSessionId();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see comment above
    setSessionId(id);
    setMessages(loadCachedMessages(id));
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    try {
      window.sessionStorage.setItem(cacheKeyFor(sessionId), JSON.stringify(messages));
    } catch {
      // Best-effort UI cache only -- losing it changes nothing about
      // correctness, since the backend remains the source of truth.
    }
  }, [messages, sessionId]);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const checkHealth = useCallback(async () => {
    const health = await getHealth();
    if (!health) {
      setConnectionStatus("offline");
    } else if (health.status === "ok") {
      setConnectionStatus("online");
    } else {
      setConnectionStatus("degraded");
    }
  }, []);

  // One check on initial load only -- never polled on an interval, per
  // the explicit "do not poll /health aggressively" requirement. This is
  // the same "fetch on mount" pattern React's own docs use as the
  // canonical example of a legitimate data-fetching effect.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see comment above
    void checkHealth();
  }, [checkHealth]);

  const callApi = useCallback(
    async (text: string) => {
      setError(null);
      setLoading(true);
      try {
        const response = await sendChatMessage({ message: text, session_id: sessionId });
        setMessages((prev) => [...prev, { role: "assistant", content: response.reply }]);
        setPendingMessage(null);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  function handleSend(text: string) {
    if (loading || !sessionId) return;
    setPendingMessage(text);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    void callApi(text);
  }

  function handleRetry() {
    if (pendingMessage && !loading) {
      void callApi(pendingMessage);
    }
  }

  function handleNewChat() {
    const id = startNewSession();
    setSessionId(id);
    setMessages([]);
    setError(null);
    setPendingMessage(null);
    setLoading(false);
  }

  return (
    <div className="flex h-dvh flex-col bg-slate-100 dark:bg-slate-950">
      <header className="flex shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-900 sm:px-6">
        <h1 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100 sm:text-base">
          AI Agent
        </h1>
        <StatusIndicator status={connectionStatus} />
      </header>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <Sidebar sessionId={sessionId} onNewChat={handleNewChat} />

        <main className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto">
            <div className="mx-auto flex min-h-full max-w-3xl flex-col gap-3 px-4 py-6 sm:px-6">
              {messages.length === 0 && !loading ? (
                <EmptyState />
              ) : (
                messages.map((message, index) => <MessageBubble key={index} message={message} />)
              )}
              {loading && <TypingIndicator />}
              {error && <ErrorBanner message={error} onRetry={pendingMessage ? handleRetry : undefined} />}
              <div ref={scrollAnchorRef} />
            </div>
          </div>

          <ChatInput onSend={handleSend} disabled={loading || !sessionId} />
        </main>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div data-testid="empty-state" className="flex flex-1 flex-col items-center justify-center py-16 text-center">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Your local AI agent</h2>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-slate-500 dark:text-slate-400">
        Ask questions, search the web when needed, and continue conversations with session
        memory.
      </p>
    </div>
  );
}

function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      data-testid="error-banner"
      className="flex items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300"
    >
      <span>{message}</span>
      {onRetry && (
        <button
          type="button"
          data-testid="retry-button"
          onClick={onRetry}
          className="shrink-0 rounded-lg border border-red-300 bg-white px-2.5 py-1 text-xs font-medium text-red-700 transition-colors hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-400/50 dark:border-red-800 dark:bg-transparent dark:text-red-300 dark:hover:bg-red-950"
        >
          Retry
        </button>
      )}
    </div>
  );
}
