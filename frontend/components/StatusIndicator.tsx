export type ConnectionStatus = "checking" | "online" | "degraded" | "offline";

const STATUS_CONFIG: Record<ConnectionStatus, { label: string; dotClass: string }> = {
  checking: { label: "Checking…", dotClass: "bg-slate-400" },
  online: { label: "Online", dotClass: "bg-emerald-500" },
  degraded: { label: "Degraded", dotClass: "bg-amber-500" },
  offline: { label: "Backend unavailable", dotClass: "bg-red-500" },
};

/**
 * A simple status dot + label, derived from GET /health's `status` field
 * only. Deliberately shows nothing more -- no schema version, no backend
 * type, no persistence detail -- an end user of the chat has no use for
 * that, and showing it would expose internal implementation state.
 */
export default function StatusIndicator({ status }: { status: ConnectionStatus }) {
  const { label, dotClass } = STATUS_CONFIG[status];

  return (
    <div data-testid="status-indicator" data-status={status} className="flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-300">
      {/* Status is communicated by the text label too, not by color alone. */}
      <span
        aria-hidden="true"
        className={`inline-block h-2 w-2 rounded-full ${dotClass} ${status === "checking" ? "animate-pulse" : ""}`}
      />
      <span role="status">{label}</span>
    </div>
  );
}
