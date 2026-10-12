# Frontend instructions

Applies to `frontend/` and descendants, in addition to the root instructions.

## Stack and structure
- This is a React 19 + TypeScript app built with Vite, Tailwind CSS v4, React Router, Axios, Motion, and Recharts.
- `src/App.tsx` defines navigation and routes. Pages are in `src/pages/`, reusable UI in `src/components/`, and shared styling in `src/index.css`.
- `src/api/client.ts` is the typed API boundary. Keep API requests relative to `/api`; `vite.config.ts` proxies `/api` and `/health` to the local backend during development. Do not hardcode a production server address in components.
- Keep response types consistent with `backend/api/schemas.py`, including score values, confidence tiers, and per-verifier details.
- Reuse existing UI components and styling conventions. Keep the existing analysis and history routes working and make loading/error behavior visible.

## Commands
From `frontend/`:
```bash
npm ci
npm run dev
npm run lint
npm run build
```
There is currently no `npm test` script. Use lint and build for the existing frontend checks; report any additional manual validation accurately.
