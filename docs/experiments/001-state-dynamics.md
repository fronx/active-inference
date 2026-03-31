# Experiment 001: Add State Dynamics to the Psychology Model

Status: proposed

Date: 2026-03-31

## Belief

The current psychology model collapses to flat lines because it has no hidden state dynamics.

All three regimes only differ in priors and precisions over hidden causes `v = [expected_energy, expected_reward, expected_fatigue]`. Since there is no `f(x, v, P)`, each time step is effectively memoryless. The model can correct belief error at one step, but it cannot carry forward depletion, recovery, or accumulated cost into the next.

If we want healthy pacing, depressive under-activation, and manic over-extension to emerge as trajectories instead of static set points, the model needs internal variables that evolve continuously.

## Action

Introduce hidden states `x` to both the generative process and generative model.

Current candidate state set:

- `x(1) = reserves` - available energy capacity
- `x(2) = fatigue` - accumulated cost / drag
- `x(3) = effort` - current expenditure level or activation state

Proposed role of each term:

- `reserves` deplete under sustained effort and recover during rest
- `fatigue` rises with exertion and decays slowly
- `effort` acts as the fast state that action pushes around, while reserves and fatigue provide slower memory

Planned process shape:

- `G(1).f`: action drives `effort`; `effort` drains `reserves`; `effort` accumulates `fatigue`
- `G(1).g`: observations remain `[energy, reward, fatigue]`, but now depend on hidden state

Planned model shape:

- `M(1).f`: mirrors the same state structure
- `M(1).g`: predicts observations from inferred `reserves`, `fatigue`, and `effort`
- regime priors continue to bias how worthwhile exertion is expected to be

## Expected

If the added state dynamics are doing the right job, we should see:

- Healthy: oscillation or at least repeated spend-rest-recover structure
- Depressed: fatigue expectation dominates, effort stays low, reward stays low, recovery is underused
- Manic: effort stays high too long, reserves deplete, reward saturates, then a crash follows

More specifically:

- `energy` should no longer be an independent per-step choice with no history
- `fatigue` should persist after expenditure instead of resetting instantly
- `reward` should become state-dependent, not just a static transform of current energy

## Observed

Current baseline result before the change:

- Healthy, depressed, and manic all flatten by about `t = 10`
- There is no genuine recovery cycle
- There is no manic overshoot and crash
- There is no depressive spiral with temporal memory

This supports the core diagnosis: the missing ingredient is stateful dynamics, not just different prior settings.

## Update

Current working belief:

- two hidden states may be enough for a better transient
- three hidden states are likely the smallest robust design if we want repeated pacing dynamics rather than just convergence to a different fixed point

The extra `effort` state is important because action in `spm_ADEM` is optimized directly. Without an internal effort state, action can collapse too quickly to a static compromise. A fast effort state gives the model inertia; reserves and fatigue then shape that trajectory over time.

## Next

1. Implement `psychology_state_f` and `psychology_state_g` for both `G` and `M`.
2. Keep observation channels near the same numeric scale so no one channel dominates through precision alone.
3. Use smooth nonlinearities only; avoid hard `abs`, `max`, or clipping inside the solver path where possible.
4. Run all three regimes and inspect whether the new dynamics produce distinct temporal signatures rather than three new fixed points.
