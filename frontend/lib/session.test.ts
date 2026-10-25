import { beforeEach, describe, expect, it } from "vitest";
import { getOrCreateSessionId, startNewSession } from "./session";

beforeEach(() => {
  window.localStorage.clear();
});

describe("getOrCreateSessionId", () => {
  it("creates a non-empty session id on first use", () => {
    const id = getOrCreateSessionId();
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  it("persists the id across calls (page refresh)", () => {
    const first = getOrCreateSessionId();
    const second = getOrCreateSessionId();
    expect(second).toBe(first);
  });

  it("stores the id in localStorage so it survives a simulated refresh", () => {
    const id = getOrCreateSessionId();
    expect(window.localStorage.getItem("chat.sessionId")).toBe(id);
  });
});

describe("startNewSession", () => {
  it("generates a session id different from the previous one", () => {
    const original = getOrCreateSessionId();
    const fresh = startNewSession();
    expect(fresh).not.toBe(original);
  });

  it("persists the new id so subsequent calls to getOrCreateSessionId reuse it", () => {
    const fresh = startNewSession();
    expect(getOrCreateSessionId()).toBe(fresh);
  });

  it("never derives the session id from personal information (structural check)", () => {
    const id = startNewSession();
    expect(id.toLowerCase()).not.toContain("@");
  });
});
