# Session Notes

## Current Focus
- React web UI for interactive psychology model exploration, backed by persistent Octave worker

## What Was Done
- **Removed categorical belief layer**: Model no longer uses softmax over archetypes (high-agency/balanced/low-agency). Hidden causes are now direct beliefs about the world: [expected_energy, expected_reward, expected_fatigue]. Categories (healthy/depressed/manic) exist only as UI presets.
- **Modularized Octave code**: Extracted from monolithic `dem_psychology.m` into `psychology/` folder with separate files: `dem_psychology_core.m`, `psychology_expect.m`, `psychology_observe.m`, `psychology_extract.m`, `psychology_params.m`
- **Built web app**: React frontend (Vite + Recharts) with parameter sliders and 4 charts. Express backend spawns persistent Octave worker via stdin/stdout protocol. Disk + client-side caching by parameter hash. SSE streaming of results.
- **Verified**: Octave JSON pipeline tested end-to-end. Both client and server type-check clean.

## Architecture
- `psychology/` — all Octave model files
- `psychology/psychology_worker.m` — persistent worker (FIFO input → stdout JSON, `__READY__`/`__DONE__` protocol)
- `web/server/` — Express backend (port 3001), spawns Octave worker on boot
- `web/client/` — Vite React app (port 5173), sliders + presets + streaming charts

## Known Issues
- Octave ignores Node.js stdin pipes (`fgetl(stdin)` never receives data). Solved by using a named pipe (FIFO) at `/tmp/octave_psychology_worker` for IPC instead.

## Next Steps
- Smoke test full stack (start both servers, run simulation through UI)
- Verify the direct-belief model produces meaningful dynamics across regimes
- Consider whether the model needs state dynamics (f function) for belief momentum
- UI polish: better loading states, parameter descriptions/tooltips

## Commands
```bash
cd web/server && npm run dev   # backend (starts Octave worker)
cd web/client && npm run dev   # frontend
```

## Key Locations
- `psychology/dem_psychology_core.m` — core simulation
- `psychology/psychology_worker.m` — persistent Octave worker
- `web/server/src/octave.ts` — worker management
- `web/client/src/components/` — UI components
- `psychology.md` — conceptual writeup
