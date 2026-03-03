# Psychology Toy Model

A minimal active inference scenario that focuses on dysregulated energy expenditure. It contrasts two pathological attractors—depression (down regulated) and mania (up regulated)—and a balanced mode. The model keeps a single control dimension (energy policy) and shows how biased feedback shapes trajectories over time.

## Structure

- **Beliefs:** `moodLogits` encodes three hypotheses (mania, balanced, depression). After a softmax, the weighted templates define the affective expectations.
- **Actions:**
  - `energyPolicy` is the effort actually spent at each step.
  - `regulationBias` represents compensatory control (medication, pacing, therapy cues, etc.).
- **Observations:**
  - `energy` reports the realized expenditure.
  - `reward` captures hedonic/goal reinforcement.
  - `fatigue` estimates metabolic cost/interoceptive drag.
  - `trajectory` is a signed feedback term that the next inference round treats as momentum (positive pushes toward mania, negative toward depression).

## Generative Model (`expect`)

Inside `src/example-psychology.ts`, `expect` mixes template values for each hidden mood:

- Energy template `[1.35, 1.00, 0.65]` encodes that mania expects high energy, depression expects low energy.
- Reward vs. fatigue templates emphasize that manic states anticipate higher reward and lower fatigue, while depression expects the opposite.
- `trajectory = mood(1) - mood(3)` summarizes preferred drift: more manic posterior probability pushes momentum upward; depressive probability pulls it down.

## Generative Process (`observe`)

The process converts actions into concrete signals with explicit feedback loops:

1. Compute deviation from a metabolic baseline (1.0) and split it into upward vs. downward excursions.
2. Upward excursions drive `mania_feedback = 0.9 * upward`, producing reinforcing reward signals.
3. Downward excursions yield `depression_feedback = 1.1 * downward` (negative), increasing fatigue and inhibition.
4. Reward is formed from the hedonic signal, deviation, and residual inhibition; fatigue grows with absolute energy plus extra penalties for downward excursions.
5. The `trajectory` observation integrates the deviation, regulation bias, differential reward-fatigue balance, and both feedback terms, so large reinforcing rewards keep pushing energy up while accumulating fatigue or negative feedback drags the system down.

This construction captures the qualitative idea that mania amplifies active energy deployment despite mounting fatigue, whereas depression suppresses energy and receives little hedonic reinforcement, making upward corrections hard.

## Running the Example

```bash
npm run example:psychology > dem_psychology.m
octave --gui dem_psychology.m
```

The first command emits MATLAB/Octave code for the toy model. Running it through SPM12's `spm_ADEM` lets you inspect how beliefs over the three modes evolve when different energy policies/regulation biases are optimized by active inference.
