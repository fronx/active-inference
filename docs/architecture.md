# DEM_morphogenesis Architecture

This document shows the flow of library functions in `DEM_morphogenesis.m`, with inputs and outputs at each stage.

## Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   CUSTOM CODE    │     │  SPM12 LIBRARY   │     │     OUTPUT       │
│   (you write)    │────▶│   (spm_ADEM)     │────▶│  (trajectories)  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

The library provides the **solver** (`spm_ADEM`). You provide:
1. **Target configuration** (P) - what the system should converge to
2. **observe()** - generative process: how the world produces observations from actions
3. **expect()** - generative model: how the agent predicts observations from beliefs
4. **morphogenesis()** - how influence/signals spread through the system

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SETUP PHASE                                       │
│                     (Define the problem domain)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DEFINE TARGET/PRIOR (P)                              [CUSTOM]           │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     Target matrix ──► p(:,:,1:4) ──► P.position    (where cells should be)  │
│         (shape)         (masks)      P.signal      (what they should emit)  │
│                                      P.sense       (what they should sense) │
│                                         │                                   │
│                           ┌─────────────┴─────────────┐                     │
│                           ▼                           ▼                     │
│                    spm_detrend()              morphogenesis()               │
│                    [LIBRARY]                     [CUSTOM]                   │
│                    centers coords            computes sensed signals        │
│                                              via spatial diffusion          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. INITIALIZE BELIEFS & ACTIONS                                            │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     identityLogits = randn(n,n)/8     (n×n matrix: "who am I?" beliefs)     │
│            │                                                                │
│            ▼                                                                │
│        expect(identityLogits, P)  ──► CellState{position, signal, sense}    │
│           [CUSTOM]                    (expected observations given beliefs) │
│            │                                                                │
│            └──► uses spm_softmax()    (converts logits to probabilities)    │
│                    [LIBRARY]                                                │
│            │                                                                │
│            ▼                                                                │
│     action.position = g.position      (initial actions match expectations)  │
│     action.signal   = g.signal                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. BUILD GENERATIVE PROCESS (G) - "The World"                              │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     R = spm_cat({...})                 (restriction matrix: which actions   │
│           [LIBRARY]                     affect which observations)          │
│                                                                             │
│     G(1).g  = @observe(x,v,action,P)   (observation function)    [CUSTOM]   │
│     G(1).v  = observe([],[],action,action)  (initial observations)          │
│     G(1).V  = exp(16)                  (high precision = low noise)         │
│     G(1).U  = exp(2)                   (action precision)                   │
│     G(1).R  = R                        (restriction matrix)                 │
│     G(1).pE = action                   (action parameters)                  │
│                                                                             │
│     G(2).a  = spm_vec(action)          (flatten action struct)   [LIBRARY]  │
│     G(2).v  = 0                        (no exogenous causes)                │
│     G(2).V  = exp(16)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. BUILD GENERATIVE MODEL (M) - "The Agent's Beliefs"                      │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     M(1).g  = @expect(x,v,P)           (observation model)       [CUSTOM]   │
│     M(1).v  = g                        (initial expectations)               │
│     M(1).V  = exp(3)                   (observation precision)              │
│     M(1).pE = P                        (prior parameters - the target!)     │
│                                                                             │
│     M(2).v  = identityLogits           (hidden causes = identity beliefs)   │
│     M(2).V  = exp(-2)                  (low precision = flexible beliefs)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. ASSEMBLE DEM STRUCTURE                                                  │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     DEM.M = M          (generative model)                                   │
│     DEM.G = G          (generative process)                                 │
│     DEM.C = zeros(1,N) (exogenous causes over time)                         │
│     DEM.U = zeros(n*n,N) (prior on hidden causes over time)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SOLVE PHASE                                       │
│                   (Run active inference)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. RUN THE SOLVER                                        [LIBRARY]         │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     DEM = spm_ADEM(DEM)                                                     │
│            │                                                                │
│            │  Internally iterates:                                          │
│            │  ┌────────────────────────────────────────────────────────┐    │
│            │  │  for t = 1:N                                           │    │
│            │  │      1. Generate observations via G(1).g (calls observe) │   │
│            │  │      2. Predict observations via M(1).g (calls expect)  │   │
│            │  │      3. Compute prediction error                       │    │
│            │  │      4. Update beliefs (identityLogits)                │    │
│            │  │      5. Update actions to minimize free energy         │    │
│            │  │      6. Store trajectory in DEM.qU, DEM.pU             │    │
│            │  └────────────────────────────────────────────────────────┘    │
│            │                                                                │
│            ▼                                                                │
│     Returns:                                                                │
│       DEM.qU.v{1}  = expected observations over time                        │
│       DEM.qU.v{2}  = identity beliefs over time (n×n×T)                     │
│       DEM.qU.a{2}  = actions over time (position & signal)                  │
│       DEM.pU.v{1}  = true observations from process                         │
│       DEM.J        = free energy over time                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. EXTRACT & VISUALIZE RESULTS                           [LIBRARY+CUSTOM]  │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     spm_DEM_qU(DEM.qU, DEM.pU)         (standard visualization)  [LIBRARY]  │
│                                                                             │
│     spm_unvec(DEM.qU.v{2}, M(2).v)     (unpack identity beliefs) [LIBRARY]  │
│           │                                                                 │
│           ▼                                                                 │
│     spm_softmax(v)                     (get probabilities)       [LIBRARY]  │
│           │                                                                 │
│           ▼                                                                 │
│     Custom plotting of cell positions, signals, trajectories     [CUSTOM]   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Custom Functions (What You Replace for New Domains)

### morphogenesis() - Signal Propagation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  morphogenesis(position, signal, y)                                         │
│  ───────────────────────────────────────────────────────────────────────    │
│  INPUT:  position (2×n) - cell positions                                    │
│          signal (m×n)   - signals emitted by each cell                      │
│          y (2×k)        - locations to sample (default: position)           │
│  OUTPUT: sense (m×k)    - signal concentration at each location             │
│                                                                             │
│  PHYSICS: sense(i) = Σⱼ exp(-distance(i,j)) × signal(j)                     │
│           (exponential decay with distance)                                 │
│                                                                             │
│  REPLACE WITH: Your domain's "influence propagation" function               │
│    - Social: influence decay with relationship distance                     │
│    - Startup: information flow through org structure                        │
│    - Psych: emotional contagion / energy transfer                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### observe() - Generative Process (World Dynamics)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  observe(x, v, action, P) - Generative Process                              │
│  ───────────────────────────────────────────────────────────────────────    │
│  INPUT:  action (struct) - current actions {position, signal}               │
│          P (struct)      - parameters                                       │
│  OUTPUT: CellState       - observed {position, signal, sense}               │
│                                                                             │
│  WHAT IT DOES: Maps actions to observations in the real world               │
│    observed.position = action.position                                      │
│    observed.signal   = action.signal                                        │
│    observed.sense    = morphogenesis(action.position, action.signal)        │
│                                                                             │
│  REPLACE WITH: Your domain's "world dynamics"                               │
│    - How does taking action X produce observation Y?                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### expect() - Generative Model (Belief-to-Prediction)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  expect(x, identityLogits, P) - Generative Model                            │
│  ───────────────────────────────────────────────────────────────────────    │
│  INPUT:  identityLogits (n×n) - beliefs about identity                      │
│          P (struct)           - prior parameters (target configuration)     │
│  OUTPUT: CellState            - expected {position, signal, sense}          │
│                                                                             │
│  WHAT IT DOES: "If I believe I'm cell k, what should I observe?"            │
│    identityBelief = softmax(identityLogits)   # who do I think I am?        │
│    expected.position = P.position × identityBelief   # where should I be?   │
│    expected.signal   = P.signal × identityBelief     # what should I emit?  │
│    expected.sense    = P.sense × identityBelief      # what should I sense? │
│                                                                             │
│  REPLACE WITH: Your domain's "belief-to-prediction" mapping                 │
│    - Given my beliefs about my role, what do I expect to experience?        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Library Functions Reference

| Function        | Purpose                                      | When Called           |
|-----------------|----------------------------------------------|-----------------------|
| `spm_ADEM()`    | THE SOLVER - runs active inference loop      | Once, after setup     |
| `spm_softmax()` | Converts logits → probabilities              | In expect(), visualization|
| `spm_vec()`     | Flatten struct → vector                      | Packing for solver    |
| `spm_unvec()`   | Vector → struct (with template)              | Unpacking results     |
| `spm_cat()`     | Concatenate cell arrays → matrix             | Building R matrix     |
| `spm_detrend()` | Center data (subtract mean)                  | Coordinate centering  |
| `spm_DEM_qU()`  | Visualize inference results                  | After solving         |
| `spm_figure()`  | Figure window management                     | Plotting              |

---

## SPM Naming Conventions

The SPM library uses single-letter variable names that follow specific conventions. We preserve these for compatibility with the library and documentation.

### Core Structures (passed to spm_ADEM)

| Name | Meaning | Mnemonic |
|------|---------|----------|
| `G` | **G**enerative process | The world / ground truth |
| `M` | Generative **M**odel | The agent's mind / beliefs |
| `DEM` | **D**ynamic **E**xpectation **M**aximization | The complete problem specification |

### Hierarchy Levels

Both `G` and `M` are arrays representing hierarchical levels:

```
G(1)  - Level 1: Maps actions → observations (calls observe())
G(2)  - Level 2: Action variables

M(1)  - Level 1: Maps beliefs → predictions (calls expect())
M(2)  - Level 2: Hidden causes (identity beliefs)
```

### Common Field Names

| Field | Meaning |
|-------|---------|
| `.g` | Observation function: `y = g(x, v, P)` |
| `.f` | State transition: `dx/dt = f(x, v, P)` (not used in morphogenesis) |
| `.v` | Hidden causes (causal states) |
| `.x` | Hidden states (not used in morphogenesis - point attractor) |
| `.V` | Precision of observations (inverse variance, higher = more confident) |
| `.W` | Precision of state transitions |
| `.U` | Precision of actions |
| `.pE` | Prior expectation of parameters |
| `.a` | Action variables |
| `.R` | Restriction matrix (which actions affect which observations) |

### Output Structures

| Name | Meaning |
|------|---------|
| `qU` | Posterior (**q**) over states (**U**) - what the agent infers |
| `pU` | Prior/Process (**p**) states (**U**) - ground truth from generative process |
| `J` | Negative free energy (log evidence) - should increase over time |

### Precision Values

Precision = 1/variance. Higher precision means more confidence/less noise.

| Value | Meaning | Typical Use |
|-------|---------|-------------|
| `exp(16)` | Very high precision | Process noise (world is deterministic) |
| `exp(3)` | Medium precision | Observation confidence |
| `exp(2)` | Medium precision | Action precision |
| `exp(-2)` | Low precision | Flexible/uncertain beliefs |

---

## Data Structures

### P (Prior/Target Configuration)
```matlab
P.position  % 2×n matrix: target [x; y] for each of n cells
P.signal    % m×n matrix: target signal emission for each cell
P.sense     % m×n matrix: expected sensed signals at target positions
```

### action (Agent Actions)
```matlab
action.position  % 2×n matrix: current [x; y] positions
action.signal    % m×n matrix: current signal emissions
```

### DEM.qU (Inference Results)
```matlab
DEM.qU.v{1}(:,t)  % Expected observations at time t (vectorized)
DEM.qU.v{2}(:,t)  % Identity belief logits at time t (vectorized n×n)
DEM.qU.a{2}(:,t)  % Actions at time t (vectorized {position, signal})
```

### DEM.J (Free Energy)
```matlab
DEM.J  % 1×T vector: free energy at each timestep (should decrease)
```
