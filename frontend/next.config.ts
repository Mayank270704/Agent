import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pins the workspace root to this directory explicitly -- without it,
  // Next.js/Turbopack auto-detects a root by walking up for a lockfile and
  // can pick an unrelated one outside this git repository (harmless, but
  // produces a startup warning every run).
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
