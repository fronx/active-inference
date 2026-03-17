# Session Notes

## Current Focus
- Building `dem_psychology.m`: active inference model of energy expenditure dysregulation (depression/mania as precision pathologies)

## Recent Context
- Model mirrors morphogenesis structure: generative process (observe) + generative model (expect) + regime-specific precisions
- Three regimes: healthy (loose priors, high sensory precision), depressed (rigid low-agency prior), manic (rigid high-agency prior)
- Depression and mania share identical precision structure — only the initial belief bias differs

## Open Threads
- Haven't run the model yet — need to verify dimensions (G(1).R = ones(3,1) with 1 action dim and 3 observables)
- Need to validate spm_ADEM integration (struct-based observations vs vector expectations)

## Commands

## Next Steps
- Run `dem_psychology('healthy')` in Octave and check for dimension errors
- Compare output plots across all three regimes
- Write up interpretation of precision pathology results

## Key Locations
- `dem_psychology.m` — main model
- `spm12/toolbox/DEM/DEM_morphogenesis.m` — reference implementation
- `spm12/spm_ADEM.m` — solver
- `psychology.md` — conceptual writeup
