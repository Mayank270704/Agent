import { expect, type Page } from "@playwright/test";

/** Shared across E2E specs: send one message and wait for the reply to render. */
export async function sendAndWaitForReply(page: Page, message: string, timeout = 45_000) {
  await page.getByTestId("chat-input").fill(message);
  await page.getByTestId("send-button").click();
  const reply = page.getByTestId("message-assistant").last();
  await expect(reply).toBeVisible({ timeout });
  // Wait for loading to clear too, so the next action on the page is safe.
  await expect(page.getByTestId("typing-indicator")).toHaveCount(0);
  return reply;
}
