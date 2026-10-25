import { test, expect } from "@playwright/test";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

const FRONTEND_ROOT = path.resolve(__dirname, "..");
const SOURCE_DIRS = ["app", "components", "lib"];

function walk(dir: string, out: string[]): void {
  for (const name of readdirSync(dir)) {
    const abs = path.join(dir, name);
    if (statSync(abs).isDirectory()) {
      walk(abs, out);
    } else if (/\.(ts|tsx)$/.test(name)) {
      out.push(abs);
    }
  }
}

function listSourceFiles(): string[] {
  const files: string[] = [];
  for (const dir of SOURCE_DIRS) {
    walk(path.join(FRONTEND_ROOT, dir), files);
  }
  return files;
}

// M. CORS -- verified as a REAL cross-origin browser request (the frontend
// runs on :3000, the backend on :8000), not simulated.
test.describe("M. CORS", () => {
  test("the actual POST /chat response carries a non-wildcard, exact-origin CORS header", async ({ page }) => {
    await page.goto("/");

    const postPromise = page.waitForResponse(
      (res) => res.url().endsWith("/chat") && res.request().method() === "POST",
    );
    await page.getByTestId("chat-input").fill("Say hi.");
    await page.getByTestId("send-button").click();
    const post = await postPromise;

    const headers = post.headers();
    const allowOrigin = headers["access-control-allow-origin"];
    expect(allowOrigin).toBe("http://localhost:3000");
    expect(allowOrigin).not.toBe("*");
    // allow_credentials=False in app/main.py -- Starlette only emits this
    // header at all when credentials are allowed, so it must be absent.
    expect(headers["access-control-allow-credentials"]).toBeUndefined();

    // The preflight OPTIONS handshake itself (which a browser may skip
    // sending if it already holds a cached, still-valid preflight result
    // for this origin/method/header combination -- Chromium's CORS
    // preflight cache, keyed by Access-Control-Max-Age) is verified
    // directly against the backend instead of relying on observing it
    // from this page, since whether the browser issues a fresh OPTIONS
    // request on any given run is a caching-timing detail, not a product
    // behavior. See docs/MILESTONE_43_E2E.md for the `curl -X OPTIONS`
    // verification of access-control-allow-methods/headers/max-age.
  });

  test("GET /health also carries a non-wildcard, exact-origin CORS header", async ({ page }) => {
    const healthPromise = page.waitForResponse((res) => res.url().endsWith("/health"));
    await page.goto("/");
    const health = await healthPromise;
    const allowOrigin = health.headers()["access-control-allow-origin"];
    expect(allowOrigin).toBe("http://localhost:3000");
    expect(allowOrigin).not.toBe("*");
  });
});

test.describe("O. Security sanity", () => {
  test("no secret-looking values or stack traces appear in the page or in network traffic during a normal flow", async ({
    page,
  }) => {
    const bodies: string[] = [];
    page.on("response", async (response) => {
      try {
        const text = await response.text();
        bodies.push(text);
      } catch {
        // Binary/empty bodies are not relevant here.
      }
    });

    await page.goto("/");
    await page.getByTestId("chat-input").fill("What time is it right now?");
    await page.getByTestId("send-button").click();
    await expect(page.getByTestId("message-assistant").last()).toBeVisible({ timeout: 45_000 });

    const pageContent = await page.content();
    const haystacks = [pageContent, ...bodies];
    const forbidden = [
      "TAVILY_API_KEY",
      "OLLAMA_BASE_URL=", // a raw .env line, not the harmless config value itself
      "Traceback (most recent call last)",
      "sk-ant-",
      "BEGIN PRIVATE KEY",
    ];

    for (const needle of forbidden) {
      for (const haystack of haystacks) {
        expect(haystack).not.toContain(needle);
      }
    }
  });

  test("no source file renders raw/unsanitized HTML via dangerouslySetInnerHTML", () => {
    for (const file of listSourceFiles()) {
      const content = readFileSync(file, "utf-8");
      expect(content, `${file} must not use dangerouslySetInnerHTML`).not.toContain("dangerouslySetInnerHTML");
    }
  });

  test("no source file reads a NEXT_PUBLIC_ env var other than the documented API base URL", () => {
    const pattern = /process\.env\.NEXT_PUBLIC_[A-Z0-9_]+/g;
    const allowed = new Set(["process.env.NEXT_PUBLIC_API_BASE_URL"]);
    for (const file of listSourceFiles()) {
      const content = readFileSync(file, "utf-8");
      const matches = content.match(pattern) ?? [];
      for (const match of matches) {
        expect(allowed.has(match), `${file} reads unexpected env var ${match}`).toBe(true);
      }
    }
  });
});
