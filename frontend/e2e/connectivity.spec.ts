import { test, expect } from "@playwright/test";
import { sendAndWaitForReply } from "./helpers";

// A. Frontend <-> API connectivity.
test.describe("A. Frontend <-> API connectivity", () => {
  test("loads successfully and shows the empty state", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "AI Agent", exact: true })).toBeVisible();
    await expect(page.getByTestId("empty-state")).toBeVisible();
    await expect(page.getByText("Your local AI agent")).toBeVisible();
  });

  test("GET /health is reachable and reflects as Online", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("status-indicator")).toHaveAttribute("data-status", "online", {
      timeout: 15_000,
    });
    await expect(page.getByTestId("status-indicator")).toContainText("Online");
  });

  test("sends a message, shows loading state, renders the reply, and clears the input", async ({ page }) => {
    await page.goto("/");
    const input = page.getByTestId("chat-input");
    await input.fill("Say the single word 'pong' and nothing else.");
    await page.getByTestId("send-button").click();

    // (3) the user's message renders immediately
    await expect(page.getByTestId("message-user").last()).toContainText("pong");
    // (6) the input is cleared right away, not after the reply arrives
    await expect(input).toHaveValue("");
    // (5) loading/typing state is visible while waiting, and the send
    // button is disabled to prevent a duplicate submission
    await expect(page.getByTestId("typing-indicator")).toBeVisible();
    await expect(page.getByTestId("send-button")).toBeDisabled();

    // (4) the assistant's reply eventually renders, and loading clears
    await expect(page.getByTestId("message-assistant").last()).toBeVisible({ timeout: 45_000 });
    await expect(page.getByTestId("typing-indicator")).toHaveCount(0);
    // The send button is disabled again only because the input was cleared
    // (nothing typed) -- not because the app is stuck. Confirm it is
    // genuinely usable by typing something new.
    await input.fill("One more message.");
    await expect(page.getByTestId("send-button")).toBeEnabled();
  });

  test("the empty state disappears once a conversation exists", async ({ page }) => {
    await page.goto("/");
    await sendAndWaitForReply(page, "Reply with exactly one word: ok.");
    await expect(page.getByTestId("empty-state")).toHaveCount(0);
  });
});
