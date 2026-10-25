type SidebarProps = {
  sessionId: string;
  onNewChat: () => void;
};

/**
 * Truncates a session_id for display only -- never used for anything
 * functional. Showing the full id adds no value for a demo user and this
 * keeps the sidebar visually compact.
 */
function shortSessionId(sessionId: string): string {
  if (sessionId.length <= 12) return sessionId;
  return `${sessionId.slice(0, 8)}…`;
}

export default function Sidebar({ sessionId, onNewChat }: SidebarProps) {
  return (
    <aside className="flex w-full shrink-0 flex-row items-center gap-3 border-b border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-900/60 md:h-full md:w-56 md:flex-col md:items-stretch md:gap-4 md:border-b-0 md:border-r md:p-4">
      <button
        type="button"
        data-testid="new-chat-button"
        onClick={onNewChat}
        className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
      >
        <span aria-hidden="true">＋</span>
        New Chat
      </button>

      <div className="hidden min-w-0 flex-1 md:block">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
          Session
        </h2>
        <p
          className="mt-1 truncate font-mono text-xs text-slate-500 dark:text-slate-400"
          title={sessionId}
        >
          {shortSessionId(sessionId) || "—"}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-slate-400 dark:text-slate-500">
          Reusing this session continues the same conversation. Start a new chat for an
          isolated conversation.
        </p>
      </div>
    </aside>
  );
}
