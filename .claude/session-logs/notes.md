# Session Notes

## [psychology-web-ui] 2026-03-31
**Focus**: Interactive browser UI for psychology active inference model
**Status**: in-progress

### Accomplished
- Removed categorical belief layer — hidden causes are now direct beliefs: [expected_energy, expected_reward, expected_fatigue]. Categories (healthy/depressed/manic) exist only as UI presets.
- Modularized Octave code into `psychology/` folder: `dem_psychology_core.m`, `psychology_expect.m`, `psychology_observe.m`, `psychology_extract.m`, `psychology_params.m`
- Built React frontend (Vite + Recharts) with parameter sliders and 4 charts. Express backend spawns persistent Octave worker. Disk + client-side caching by parameter hash. SSE streaming.
- Fixed Octave IPC — switched from stdin pipe to named FIFO at `/tmp/octave_psychology_worker`
- Adapted session-start skill for slow-pace projects: briefing shows last 3 active days (not calendar days), done-stream prune threshold extended to 2 weeks
- Replaced preset dropdown with one-click button row (auto-runs simulation on click)
- Synced web UI data pipeline with new state-space model: added reserves, effort, fatigueState, actionTarget traces through server types, SSE streaming, client types, and charts
- Updated chart panels to match dem_psychology.m: "State-gated expenditure" and "Outcomes and cost memory"
- Cleared stale simulation cache (old schema)
- Auto-run healthy preset on first load
- URL-based history: params encoded in URL (`?p=healthy` or `?b=...&pv=...&sv=...&n=...`), browser back/forward restores and re-runs
- Added activation and opportunity traces to UI pipeline

### Open Threads
- `web/client/README.md` still has Vite boilerplate — replace or delete
- Preset param values in presets.ts and server index.ts still old — need updating when psychology_params.m changes

### Next Steps
- Smoke test full stack with the new state-space model
- Update preset values once regime parameters are finalized

### Key Locations
- `psychology/dem_psychology_core.m` — core simulation (now with hidden states)
- `psychology/psychology_defaults.m` — shared parameters and initial conditions
- `psychology/psychology_worker.m` — persistent Octave worker
- `web/server/src/octave.ts` — worker management
- `web/client/src/components/` — UI components
- `.claude/skills/session-start/SKILL.md` — session start skill
