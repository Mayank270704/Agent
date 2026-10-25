import type { Message } from "@/lib/types";

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div
      data-testid={isUser ? "message-user" : "message-assistant"}
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words shadow-sm ${
          isUser
            ? "bg-indigo-600 text-white rounded-br-sm"
            : "bg-white text-slate-900 border border-slate-200 rounded-bl-sm dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700"
        }`}
      >
        <span className="sr-only">{isUser ? "You said: " : "Agent replied: "}</span>
        {message.content}
      </div>
    </div>
  );
}
