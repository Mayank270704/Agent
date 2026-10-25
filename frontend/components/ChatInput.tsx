"use client";

import { useRef, useState, type KeyboardEvent } from "react";

// Mirrors app/main.py's MAX_MESSAGE_LENGTH -- gives the user immediate
// feedback instead of a round trip to the backend for an oversized
// message. Kept as a local constant (no shared schema file exists between
// frontend and backend); must be updated together with the backend value
// if that ever changes.
const MAX_MESSAGE_LENGTH = 8000;

type ChatInputProps = {
  onSend: (text: string) => void;
  disabled: boolean;
};

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    textareaRef.current?.focus();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
    // Shift+Enter falls through to the textarea's own default behavior
    // (insert a newline) -- nothing to do here.
  }

  const remaining = MAX_MESSAGE_LENGTH - value.length;
  const nearLimit = remaining <= 200;

  return (
    <div className="border-t border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900 sm:p-4">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <label htmlFor="chat-message-input" className="sr-only">
          Message the agent
        </label>
        <textarea
          id="chat-message-input"
          data-testid="chat-input"
          ref={textareaRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          maxLength={MAX_MESSAGE_LENGTH}
          placeholder="Ask anything…"
          rows={1}
          className="max-h-40 min-h-[2.75rem] flex-1 resize-none rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:placeholder:text-slate-500"
        />
        <button
          type="button"
          data-testid="send-button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="inline-flex h-11 shrink-0 items-center justify-center rounded-xl bg-indigo-600 px-4 text-sm font-medium text-white transition-colors hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500 dark:disabled:bg-slate-700 dark:disabled:text-slate-400"
        >
          {disabled ? "Sending…" : "Send"}
        </button>
      </div>
      <div className="mx-auto mt-1.5 flex max-w-3xl items-center justify-between px-0.5 text-xs text-slate-400 dark:text-slate-500">
        <span>Enter to send · Shift+Enter for a new line</span>
        {nearLimit && <span className={remaining < 0 ? "text-red-500" : ""}>{remaining} characters left</span>}
      </div>
    </div>
  );
}
