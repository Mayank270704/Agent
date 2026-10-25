import { test, expect } from "@playwright/test";
import { sendAndWaitForReply } from "./helpers";

/**
 * E. Explicit web search -- deliberately exactly ONE test in this entire
 * E2E suite. Tavily has a limited free quota; every other scenario in
 * this suite uses the time/date tools or plain LLM answers, which cost
 * nothing. Do not add a second web-search test without a concrete reason.
 */
test.describe("E. Explicit web search", () => {
  test("an explicit, qualified search command reaches the agent and returns an answer", async ({ page }) => {
    await page.goto("/");
    const reply = await sendAndWaitForReply(
      page,
      "Search the web for the current version number of the Next.js framework.",
      60_000, // a real web_search call adds real network latency on top of the LLM call
    );
    const text = ((await reply.textContent()) ?? "").trim();
    expect(text.length).toBeGreaterThan(0);
    // The agent must not announce it can't search -- that would mean the
    // tool was not actually invoked/available (e.g. a missing API key).
    expect(text.toLowerCase()).not.toContain("i don't have access to the internet");
    expect(text.toLowerCase()).not.toContain("i cannot browse");
  });
});
