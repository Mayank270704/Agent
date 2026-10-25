import { test, expect, type Route } from "@playwright/test";

test.describe("I. Invalid input handling", () => {
  test("an empty input cannot be submitted", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("send-button")).toBeDisabled();
    await expect(page.getByTestId("message-user")).toHaveCount(0);
  });

  test("a whitespace-only input cannot be submitted", async ({ page }) => {
    await page.goto("/");
    const input = page.getByTestId("chat-input");
    await input.fill("   ");
    await expect(page.getByTestId("send-button")).toBeDisabled();

    // Enter must not bypass the same guard.
    await input.press("Enter");
    await expect(page.getByTestId("message-user")).toHaveCount(0);
  });

  test("an oversized input is capped client-side at the backend's own limit", async ({ page }) => {
    await page.goto("/");
    const input = page.getByTestId("chat-input");
    const oversized = "a".repeat(9000);
    await input.fill(oversized);
    const actual = await input.inputValue();
    // Mirrors app/main.py's Field(max_length=8000) -- the textarea's own
    // maxLength attribute prevents ever constructing an over-limit request.
    expect(actual.length).toBeLessThanOrEqual(8000);
  });
});

test.describe("J. Backend-unavailable handling", () => {
  test("a network failure shows a safe error, clears loading, and Retry recovers", async ({ page }) => {
    await page.goto("/");

    await page.route("**/chat", (route: Route) => route.abort("failed"));

    const input = page.getByTestId("chat-input");
    await input.fill("Hello?");
    await page.getByTestId("send-button").click();

    const banner = page.getByTestId("error-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText(/unable to reach the agent/i);

    // (no raw exception/stack trace leaked to the user)
    await expect(banner).not.toContainText(/traceback|typeerror|fetch failed/i);

    // The UI must not be stuck: loading clears and sending is possible again.
    await expect(page.getByTestId("typing-indicator")).toHaveCount(0);
    await expect(page.getByTestId("retry-button")).toBeVisible();
    // The send button is disabled only because the input was cleared on
    // submit (nothing typed) -- not because the app is stuck.
    await input.fill("Still usable?");
    await expect(page.getByTestId("send-button")).toBeEnabled();

    // Recovery: once the backend is reachable again, Retry succeeds.
    await page.unroute("**/chat");
    await page.getByTestId("retry-button").click();
    await expect(page.getByTestId("message-assistant").last()).toBeVisible({ timeout: 45_000 });
    await expect(page.getByTestId("error-banner")).toHaveCount(0);
  });
});

test.describe("K. HTTP error-status handling", () => {
  test("a 422 validation error surfaces the backend's own safe message, not raw JSON", async ({ page }) => {
    await page.goto("/");
    await page.route("**/chat", (route: Route) =>
      route.fulfill({
        status: 422,
        contentType: "application/json",
        body: JSON.stringify({
          detail: [{ loc: ["body", "message"], msg: "String should have at most 8000 characters", type: "string_too_long" }],
        }),
      }),
    );

    await page.getByTestId("chat-input").fill("Trigger a validation error.");
    await page.getByTestId("send-button").click();

    const banner = page.getByTestId("error-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("String should have at most 8000 characters");
    await expect(banner).not.toContainText(/\[\{"loc"/);
  });

  test("a 500 error surfaces a safe generic message, never a stack trace", async ({ page }) => {
    await page.goto("/");
    await page.route("**/chat", (route: Route) =>
      route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Internal server error." }),
      }),
    );

    await page.getByTestId("chat-input").fill("Trigger a server error.");
    await page.getByTestId("send-button").click();

    const banner = page.getByTestId("error-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("Internal server error.");
    await expect(banner).not.toContainText(/traceback|raise |line \d+, in /i);
  });
});

test.describe("L. Malformed-response handling", () => {
  test("a 200 response with a non-JSON body is treated as an unexpected response, not a crash", async ({ page }) => {
    await page.goto("/");
    await page.route("**/chat", (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: "text/plain",
        body: "not json at all",
      }),
    );

    const input = page.getByTestId("chat-input");
    await input.fill("Trigger a malformed response.");
    await page.getByTestId("send-button").click();

    const banner = page.getByTestId("error-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText(/unexpected response/i);
    // The page itself must still be usable afterwards.
    await input.fill("Still usable?");
    await expect(page.getByTestId("send-button")).toBeEnabled();
  });

  test("a 200 response missing the reply field is treated as an unexpected response", async ({ page }) => {
    await page.goto("/");
    await page.route("**/chat", (route: Route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ok: true }),
      }),
    );

    await page.getByTestId("chat-input").fill("Trigger a shape-mismatched response.");
    await page.getByTestId("send-button").click();

    const banner = page.getByTestId("error-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText(/unexpected response/i);
  });
});
