# Frontend

A minimal chat UI for the AI agent backend in `../app/`. Next.js (App Router) +
React + TypeScript + Tailwind CSS.

See the repository root `README.md` for the full setup (backend + frontend
together, ports, CORS). Quick start, assuming the backend is already running on
`http://localhost:8000`:

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Scripts

- `npm run dev` — development server
- `npm run build` — production build (also runs the TypeScript check)
- `npm run start` — serve the production build
- `npm run lint` — ESLint
- `npm run typecheck` — TypeScript check only, no build
- `npm test` — focused unit tests (`lib/`), via Vitest

## Configuration

One environment variable: `NEXT_PUBLIC_API_BASE_URL` (see `.env.local.example`).
Defaults to `http://localhost:8000` if unset — no `.env.local` is required for
local development against the default backend setup.

## Structure

- `app/` — Next.js App Router entry (`layout.tsx`, `page.tsx`)
- `components/` — UI components (`ChatApp` is the orchestrator)
- `lib/` — API client, session-id handling, shared types (no UI code)

See `../docs/MILESTONE_42_FRONTEND.md` for the full architecture writeup.
