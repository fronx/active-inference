# Active Inference with SPM: From First Principles

This tutorial builds understanding from the core interface outward.

## The Heart: spm_ADEM

Everything revolves around one function:

```matlab
DEM = spm_ADEM(DEM)
```

**Input:** A structure containing your problem specification
**Output:** The same structure, enriched with the solution

Think of it as a black box that solves active inference problems. You don't need to understand the math inside - you just need to know what to feed it and what you get back.

---

## The Interface Contract

### What You Give spm_ADEM

```matlab
DEM.M  % Model (agent's beliefs about the world)
DEM.G  % Process (the actual world dynamics)
DEM.C  % Causes (external inputs over time)
DEM.U  % Prior expectations on causes
```

### What spm_ADEM Returns

The same `DEM` structure, plus:

```matlab
% Ground truth (what actually happened in the world)
DEM.pU.v{1}  % Observations at each timestep (vectorized)
DEM.pU.v{2}  % True hidden causes at each timestep
DEM.pU.x     % True hidden states (if using state dynamics)

% Inferred quantities (what the agent believes)
DEM.qU.v{1}  % Expected observations at each timestep
DEM.qU.v{2}  % Inferred hidden causes (beliefs) at each timestep
DEM.qU.a{2}  % Actions taken at each timestep
DEM.qU.z     % Prediction errors

% Meta
DEM.F        % Free energy over time (should decrease)
```

**Key insight:**
- `pU` = **p**rocess truth (the world)
- `qU` = **q**ueried/inferred (the agent's beliefs)

The agent tries to make `qU` match `pU` by updating beliefs and taking actions.

---

## What is M? (The Model)

`M` describes **how the agent thinks the world works**.

It's a hierarchical structure. For a 2-level model:

```matlab
% Level 1: Observations
M(1).g  = @function(x, v, P)  % "How do hidden causes produce observations?"
M(1).v  = initial_guess        % Starting expectations
M(1).V  = exp(3)               % Precision of observations
M(1).pE = parameters           % Prior parameters

% Level 2: Hidden causes (beliefs)
M(2).v  = initial_beliefs      % Starting belief state
M(2).V  = exp(-2)              % Precision of beliefs (how flexible?)
```

**M(1).g is your generative model** - the function that says:
> "Given beliefs `v`, what observations do I expect?"

---

## What is G? (The Process)

`G` describes **how the world actually works**.

```matlab
% Level 1: How actions generate observations
G(1).g  = @function(x, v, a, P)  % "How do actions produce observations?"
G(1).v  = initial_observations   % Starting observations
G(1).V  = exp(16)                % Precision (world is deterministic)
G(1).U  = exp(2)                 % Action precision
G(1).R  = restriction_matrix     % Which actions affect which observations
G(1).pE = action_parameters      % Parameters (e.g., positions, signals)

% Level 2: Action variables
G(2).a  = initial_actions        % Starting actions (vectorized)
G(2).v  = 0                      % No exogenous causes
G(2).V  = exp(16)                % High precision
```

**G(1).g is your generative process** - the function that says:
> "Given actions `a`, what observations actually occur?"

---

## What are C and U? (Exogenous Inputs)

```matlab
DEM.C = zeros(1, N)      % External causes over N timesteps (usually none)
DEM.U = zeros(n, N)      % Prior expectations on causes over time
```

For most simulations (including morphogenesis), these are zeros - the system is autonomous with no external driving forces.

---

## The Active Inference Loop (What spm_ADEM Does)

```
for t = 1:N timesteps
    1. Generate observations: pU.v{1}(t) = G(1).g(actions)     [PROCESS]
    2. Predict observations:  qU.v{1}(t) = M(1).g(beliefs)     [MODEL]
    3. Compute error:         error = predicted - observed
    4. Update beliefs:        minimize free energy w.r.t. beliefs
    5. Update actions:        minimize free energy w.r.t. actions
    6. Store trajectory
end
```

The agent simultaneously:
- **Updates beliefs** to better explain observations
- **Updates actions** to make observations match expectations

Both updates reduce **free energy** (prediction error).

---

## Next: Building Your First Model

Now that we know the interface, we can work backwards to specify M and G for any domain.

The key questions are:
1. What are my **observations**? (what can be sensed)
2. What are my **beliefs**? (hidden causes)
3. What are my **actions**? (how the agent affects the world)
4. How do beliefs → predictions? (M(1).g)
5. How do actions → observations? (G(1).g)

We'll build a minimal example next.
