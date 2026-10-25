import { test, expect } from "@playwright/test";
import { sendAndWaitForReply } from "./helpers";

const SESSION_STORAGE_KEY = "chat.sessionId";

test.describe("F. Multi-turn session (conversation memory)", () => {
  test("a second message in the same session sees the first message's context", async ({ page }) => {
    await page.goto("/");

    await sendAndWaitForReply(page, "My favorite programming language is Python. Just acknowledge briefly.");
    const reply2 = await sendAndWaitForReply(
      page,
      "Repeat back exactly which programming language I just mentioned, nothing else.",
      45_000,
    );
    const text2 = ((await reply2.textContent()) ?? "").toLowerCase();

    expect(text2).toContain("python");

    // This is conversation memory (ConversationMemory), not semantic
    // memory -- SEMANTIC_MEMORY_ENABLED is off by default and this test
    // does not set it. See docs/MILESTONE_43_E2E.md for the distinction.
  });
});

test.describe("G. New Chat isolation", () => {
  test("New Chat starts an isolated session with a different session_id", async ({ page }) => {
    await page.goto("/");

    await sendAndWaitForReply(page, "My favorite color is teal. Just acknowledge briefly.");
    const sessionA = await page.evaluate(
      (key) => window.localStorage.getItem(key),
      SESSION_STORAGE_KEY,
    );
    expect(sessionA).toBeTruthy();

    await page.getByTestId("new-chat-button").click();

    // Visible transcript is cleared immediately.
    await expect(page.getByTestId("message-user")).toHaveCount(0);
    await expect(page.getByTestId("message-assistant")).toHaveCount(0);
    await expect(page.getByTestId("empty-state")).toBeVisible();

    const sessionB = await page.evaluate(
      (key) => window.localStorage.getItem(key),
      SESSION_STORAGE_KEY,
    );
    expect(sessionB).toBeTruthy();
    expect(sessionB).not.toBe(sessionA);

    const reply = await sendAndWaitForReply(
      page,
      "What is my favorite color? If you don't know, say you don't know.",
    );
    const text = ((await reply.textContent()) ?? "").toLowerCase();

    // Session B must NOT have inherited session A's conversation.
    expect(text).not.toContain("teal");
  });
});

test.describe("H. Refresh persistence", () => {
  test("the session_id and conversation survive a page refresh", async ({ page }) => {
    await page.goto("/");

    await sendAndWaitForReply(page, "My favorite number is 17. Just acknowledge briefly.");
    const sessionBefore = await page.evaluate(
      (key) => window.localStorage.getItem(key),
      SESSION_STORAGE_KEY,
    );
    expect(sessionBefore).toBeTruthy();

    await page.reload();

    // (3) same session_id remains in localStorage after refresh
    const sessionAfter = await page.evaluate(
      (key) => window.localStorage.getItem(key),
      SESSION_STORAGE_KEY,
    );
    expect(sessionAfter).toBe(sessionBefore);

    // The cached transcript (sessionStorage UI cache) also survives a
    // same-tab refresh.
    await expect(page.getByTestId("message-user").first()).toContainText("17");

    // (4/5) continuing the conversation reaches the backend with the SAME
    // session_id -- verified by confirming the backend still recalls
    // context from before the refresh.
    const reply = await sendAndWaitForReply(
      page,
      "Repeat back exactly which number I mentioned earlier, nothing else.",
      45_000,
    );
    const text = (await reply.textContent()) ?? "";
    expect(text).toContain("17");
  });
});
