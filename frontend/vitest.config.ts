import path from "node:path";
import { defineConfig } from "vitest/config";

/**
 * Focused unit tests only (lib/api.ts, lib/session.ts): pure request/
 * response/error-handling and session-id logic, no component rendering,
 * no E2E browser automation. Deliberately minimal -- this project had no
 * frontend testing framework before this milestone; a full React Testing
 * Library / Playwright stack is left for Step 43 (Full Product
 * End-to-End Testing) rather than added here "merely for this milestone."
 */
export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["lib/**/*.test.ts"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
