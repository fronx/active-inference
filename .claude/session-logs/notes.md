# Session Notes

## [psychology-web-ui] 2026-03-31
**Focus**: Interactive browser UI for psychology active inference model
**Status**: in-progress

### Accomplished
- Removed categorical belief layer — hidden causes are now direct beliefs: [expected_energy, expected_reward, expected_fatigue]. Categories (healthy/depressed/manic) exist only as UI presets.
- Modularized Octave code into `psychology/` folder: `dem_psychology_core.m`, `psychology_expect.m`, `psychology_observe.m`, `psychology_extract.m`, `psychology_params.m`
- Built React frontend (Vite + Recharts) with parameter sliders and 4 charts. Express backend spawns persistent Octave worker. Disk + client-side caching by parameter hash. SSE streaming.
- Fixed Octave IPC — switched from stdin pipe to named FIFO at `/tmp/octave_psychology_worker`
- Octave JSON pipeline tested end-to-end. Both client and server type-check clean.
- Documented web UI setup in project README.md
- Adapted session-start skill for slow-pace projects: briefing shows last 3 active days (not calendar days), done-stream prune threshold extended to 2 weeks

### Open Threads
- `web/client/README.md` still has Vite boilerplate — replace or delete
- Does the model need state dynamics (f function) for belief momentum?

### Next Steps
- Smoke test full stack (start both servers, run simulation through UI)
- Verify the direct-belief model produces meaningful dynamics across regimes
- UI polish: better loading states, parameter descriptions/tooltips

### Key Locations
- `psychology/dem_psychology_core.m` — core simulation
- `psychology/psychology_worker.m` — persistent Octave worker
- `web/server/src/octave.ts` — worker management
- `web/client/src/components/` — UI components
- `psychology.md` — conceptual writeup
- `.claude/skills/session-start/SKILL.md` — session start skill
