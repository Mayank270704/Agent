import { defineConfig, devices } from "@playwright/test";

/**
 * Step 43 — Full Product End-to-End Testing.
 *
 * These tests drive a REAL browser (Chromium) against the REAL frontend
 * dev server AND the real backend (FastAPI + real Ollama + real Tavily,
 * for the one deliberately minimal web-search test). The backend is NOT
 * started by this config -- it has its own toolchain (Python/uvicorn) and
 * must already be running at http://localhost:8000 before `npm run
 * e2e` -- see docs/MILESTONE_43_E2E.md and the root README's "Running the
 * application" section for the exact command. The frontend dev server
 * IS started automatically below (and reused if already running), since
 * that part of the stack shares this project's own toolchain.
 *
 * `workers: 1` is deliberate, not a default: these tests share one real
 * backend process and must not race each other for Ollama's attention
 * (every real LLM call already takes several seconds) or make Tavily
 * calls concurrently.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
