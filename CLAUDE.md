# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project reproduces morphogenesis simulations from Friston et al. 2015 ("Knowing one's place: a free-energy approach to pattern regulation"). 16 cells start at identical positions and self-assemble into a target morphology (head-body-tail) over 32 time steps using active inference (free energy minimization).

**Goal:** Understand `spm12/toolbox/DEM/DEM_morphogenesis.m` deeply enough to create active inference models in other domains:
- Psychological phenomena (first-person energy dynamics)
- Cult dynamics (unifying/self-policing forces)
- Relationship dynamics (mutual investment patterns)
- Startup growth (specialization, alignment as free energy minimization)

## Running the Simulation

```bash
# Requires GNU Octave: brew install octave
octave --gui run_morphogenesis.m
```

This runs the morphogenesis demo, pauses for figure inspection, then generates `morphogenesis.mp4` from saved frames.

## Architecture

### Entry Point
- `run_morphogenesis.m` - Sets up paths and runs `DEM_morphogenesis`

### Core Algorithm
- `spm12/toolbox/DEM/DEM_morphogenesis.m` - Main morphogenesis simulation
  - Defines target morphology as a 2D grid with cell types (0-4 encoding RGB signals)
  - Sets up generative process (G) and generative model (M) hierarchies
  - Calls `spm_ADEM()` to run active inference

### Key SPM12 Components
- `spm12/spm_ADEM.m` - Active Dynamic Expectation Maximization solver. Integrates generative process and model inversion in parallel, enabling active inference with action variables
- `spm12/toolbox/DEM/` - DEM toolbox containing demos and supporting functions
- `spm12/spm_softmax.m`, `spm_vec.m`, `spm_unvec.m`, `spm_detrend.m` - Utility functions used throughout

### DEM Structure (passed to spm_ADEM)
```matlab
DEM.M  % Generative model hierarchy (recognition model)
DEM.G  % Generative process hierarchy (world model with actions)
DEM.C  % Causes (exogenous inputs)
DEM.U  % Prior expectations on causes
```

### Model Hierarchy (M and G)
Each level specifies:
- `.g` - Observation function: y = g(x,v,P)
- `.f` - State transition: dx/dt = f(x,v,P)
- `.V` - Precision (noise)
- `.pE` - Prior expectations

## Key Concepts

**Active Inference:** Agents minimize variational free energy - the gap between predictions/beliefs and sensory evidence. In morphogenesis, each cell infers its identity relative to others and behaves accordingly.

**Identity Logits:** Hidden causes representing each cell's belief about which target position it corresponds to. Transformed via softmax to get identity beliefs.

**Morphogenesis Function:** Computes signal concentration at each position based on distance-weighted sum of all cell signals (exponential decay with distance).

## Notes

- `spm12/spm_platform.m` was patched to support Apple Silicon (arm64)
- Output frames saved to `morphogenesis_frames/`, video to `morphogenesis.mp4`
- SPM12 source: https://github.com/spm/spm12