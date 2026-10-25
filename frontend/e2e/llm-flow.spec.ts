import { test, expect } from "@playwright/test";
import { sendAndWaitForReply } from "./helpers";

/**
 * B/C/D: proves the full browser->reply experience for a static question,
 * an explicit time query, and an explicit date query. This proves the
 * USER-VISIBLE pipeline; it deliberately does NOT assert which tool (if
 * any) ran internally -- the public API never exposes that (by design,
 * the security boundary never echoes internal execution detail), so that
 * is verified separately, directly against the backend, in
 * docs/MILESTONE_43_E2E.md's "Automated backend-internal verification"
 * section, reusing the existing deterministic-routing regression suite
 * (tests/test_deterministic_temporal_routing.py) plus one direct
 * real-Ollama check -- not re-implemented here.
 */
test.describe("B. Normal static-knowledge flow (no tool expected)", () => {
  test("answers a static question directly", async ({ page }) => {
    await page.goto("/");
    const reply = await sendAndWaitForReply(
      page,
      "In one short sentence, what is the capital of France?",
    );
    const text = (await reply.textContent()) ?? "";
    expect(text.toLowerCase()).toContain("paris");
  });
});

test.describe("C. Time tool", () => {
  test("an explicit current-time query gets a real, non-empty answer", async ({ page }) => {
    await page.goto("/");
    const reply = await sendAndWaitForReply(page, "What time is it right now?");
    const text = ((await reply.textContent()) ?? "").trim();
    expect(text.length).toBeGreaterThan(0);
    expect(text.toLowerCase()).not.toContain("i don't have access");
    expect(text.toLowerCase()).not.toContain("i cannot");
  });
});

test.describe("D. Date tool", () => {
  test("an explicit current-date query gets a real, non-empty answer", async ({ page }) => {
    await page.goto("/");
    const reply = await sendAndWaitForReply(page, "What is today's date?");
    const text = ((await reply.textContent()) ?? "").trim();
    expect(text.length).toBeGreaterThan(0);
    expect(text.toLowerCase()).not.toContain("i don't have access");
  });
});
