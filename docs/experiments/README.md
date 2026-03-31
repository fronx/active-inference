# Experiments

This folder treats model-building itself as an active inference loop.

Each experiment starts with a belief about how the model works or why it is failing. We act on that belief by changing code, priors, precisions, or equations. Reality answers by producing trajectories, free-energy traces, and failure modes. We then update the belief and choose the next intervention.

The loop is:

1. Belief: a concrete hypothesis about the model
2. Action: a code change or parameter intervention
3. Observation: the curves or solver behavior we actually get
4. Update: what we now believe, and what to try next

## Why keep this log

- It makes modeling decisions legible instead of burying them in commits.
- It separates expected behavior from actual behavior.
- It gives each iteration a falsifiable statement.
- It creates a history of how the model learned from resistance.

## Experiment format

Each experiment file should capture:

- `Status`: proposed, running, completed, abandoned
- `Belief`: the hypothesis behind the change
- `Action`: the code or model intervention
- `Expected`: what curves or qualitative regimes should appear
- `Observed`: what actually happened after running it
- `Update`: what changed in our understanding
- `Next`: the next experiment that follows from the result

## Index

- [001-state-dynamics.md](/Users/fnx/code/active-inference/docs/experiments/001-state-dynamics.md) - Add hidden states so the psychology model can remember exertion over time
- [002-opportunity-pacing.md](/Users/fnx/code/active-inference/docs/experiments/002-opportunity-pacing.md) - Reframe the model around opportunity pulses, activation, reserves, and recovery pacing
- [003-body-self-model-split.md](/Users/fnx/code/active-inference/docs/experiments/003-body-self-model-split.md) - Separate body reality from self-model beliefs and add slow capacity dynamics
- [004-felt-energy-observation.md](/Users/fnx/code/active-inference/docs/experiments/004-felt-energy-observation.md) - Separate felt energy from realized output and reward in the observation model
- [005-body-limited-mobilization.md](/Users/fnx/code/active-inference/docs/experiments/005-body-limited-mobilization.md) - Make action depend on bodily leverage and retune capacity adaptation
- [006-felt-energy-damping.md](/Users/fnx/code/active-inference/docs/experiments/006-felt-energy-damping.md) - Make subjective vitality depend more strongly on bodily availability
