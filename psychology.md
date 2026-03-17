# Active Inference <> Psychology

## Active Inference in Brief

An agent has two models running in parallel:

**Generative process (G)** — how the world actually works. Takes the agent's actions, produces real observations. The agent can't see inside this; it only sees the output.

**Generative model (M)** — the agent's theory of how the world works. Takes hidden causes (beliefs), predicts what observations *should* look like.

The gap between predicted and actual observations is **prediction error**. The agent minimizes it two ways simultaneously:

- **Perception**: update beliefs to better predict what's already being observed
- **Action**: change behavior to make observations match predictions

**Precisions** weight how much each signal matters. High sensory precision = trust observations. High prior precision = trust existing beliefs. The ratio between them determines whether new evidence can update the agent's model or gets ignored.

```
        Model (M)                             Process (G)
        the agent's beliefs                   reality

              belief logits                    action
               M(2).v                          G(2).a
                   |                               |
   DEM.U ─ pi_b ─>|  softmax + templates           | physics
   (attractor)     |                               |
                   v                               v
              expect()                        observe()
                   |                               |
                   v                               v
             predicted obs -----> PE <----- actual obs
                                  |
                                 pi_s
                                  |
                        +---------+---------+
                        |                   |
                   perception            action
                   update beliefs        update energy
                        |                   |
                        +-----> loop <------+
```

The solver (`spm_ADEM`) runs this loop forward in time, updating beliefs and actions at each step to minimize variational free energy — the agent's best estimate of how wrong its model is.

## The Model

An agent expends energy and observes what comes back: reward (diminishing returns) and fatigue (quadratic cost). It maintains core beliefs about the relationship between effort and outcomes — beliefs that shape how much energy it expends, which shapes what it observes, which reinforces the beliefs.

The agent doesn't know it's "depressed" or "manic." It just thinks it's being realistic. The labels are ours, not the agent's. From inside, each regime feels like an accurate read on how the world works. What the model actually represents are core beliefs about self-efficacy:

- "Effort pays off. I am capable. Everything is worth my energy."
- "Returns are moderate. Some things are worth effort."
- "Effort is futile. I am ineffective. Nothing is worth my energy."

These aren't mood states — they're the implicit self-model that mood is downstream of. The softmax over logits gives a weighted blend of these beliefs, and the blend determines behavior.

In this toy model, reality is deliberately simple and symmetric — so that the asymmetries in behavior emerge entirely from precision pathology, not from the physics.

### Three timescales of inference

**Perception** (fast): "How does effort relate to outcomes?"
Core belief logits updated via prediction error. Softmax gives belief weights.

    M(2).v = belief logits — where inference starts (initial value)
    DEM.U  = prior expectation on belief logits over time (the attractor)

Key distinction: M(2).v is the starting point. DEM.U is what the system is pulled toward. M(2).V controls how strongly. In morphogenesis, both default to zero because cells discover identity from scratch. In psychology, DEM.U encodes the chronic self-model — the agent's settled core belief.

**Attention** (medium): "How much should I trust my senses vs. my beliefs?"
Precision balance determines whether evidence can update beliefs.

    M(1).V = sensory precision (π_s) — trust in observations
    M(2).V = prior precision (π_b) — how rigid the attractor is

**Learning** (slow): "What do these states even feel like?"
Templates defining each belief state's expected observations. These are learned associations, not fixed truths.

    M(1).pE = templates (prior expectations on parameters)
    M(1).pC = template flexibility (prior covariance — how learnable)

### Generative process (reality)

Simple physics. No psychology baked in.

    reward  = energy / (1 + |energy|)    diminishing returns
    fatigue = 0.3 * energy^2             quadratic cost

### Generative model (the agent's beliefs)

Templates define what each core belief predicts in terms of observables:

               "effort       "moderate     "effort
                pays off"     returns"      is futile"
    energy     1.4           1.0           0.6
    reward     0.8           0.5           0.2
    fatigue    0.2           0.4           0.8

Expected observation = template * softmax(belief logits).

The moderate-returns template roughly matches reality at moderate energy. The extreme templates don't — "effort pays off" overestimates reward; "effort is futile" overestimates fatigue.

## Dysregulation

When the precision balance prevents the model from converging to reality.

The mechanism is the same in both directions: the agent's actions produce the evidence that confirms its beliefs.

An agent believing effort is futile produces anti-agency evidence (low effort → low reward → "see, it doesn't work").

An agent believing effort always pays off produces pro-agency evidence (high effort → some reward → "see, everything works").

Neither is passively misreading reality — they are actively generating a biased evidence base through their own behavior.

### Regulated (healthy)

    DEM.U  = [0; 0; 0]     flat prior — no chronic bias
    M(2).V = exp(-2)        loose prior — beliefs update from evidence
    M(1).V = exp(3)         high sensory precision — trusts observations
    M(1).pC > 0             templates are revisable

The agent discovers that moderate effort minimizes free energy because the moderate template most closely matches what reality actually delivers. If the templates are slightly wrong, they update.

### Dysregulated down (~ depression)

    DEM.U  = [-1; 0; 2]    chronic attractor: "effort is futile"
    M(2).v = [-1; 0; 2]    starts at the attractor
    M(2).V = exp(4)         rigid prior — core belief strongly held
    M(1).V = exp(0)         low sensory precision — discounts observations
    M(1).pC ~ 0             templates frozen

The agent believes effort is futile, acts with low energy, gets low reward (because reality IS low-reward at low effort), and sees this as confirmation. Self-fulfilling prophecy operating through free energy minimization.

The double bind: not only does the agent refuse to update its core belief, it also stops updating what the alternatives would feel like. The moderate template degrades from disuse.

### Dysregulated up (~ mania)

    DEM.U  = [2; 0; -1]    chronic attractor: "effort always pays off"
    M(2).v = [2; 0; -1]    starts at the attractor
    M(2).V = exp(4)         rigid prior
    M(1).V = exp(0)         discounts evidence
    M(1).pC ~ 0             templates frozen

The agent believes everything it does works, expends high energy, hits diminishing returns and rising fatigue cost, but ignores the prediction errors. Unstable — eventually the mismatch between expected and actual outcomes becomes too large to suppress.

### Therapy as precision intervention

Therapy doesn't push the agent to a new state directly. It changes the conditions under which the agent can update itself:

1. Shift DEM.U toward [0;0;0] — weaken the chronic core belief ("your story about yourself is not the only possibility")
2. Decrease M(2).V — loosen the belief's grip ("your belief about yourself is a hypothesis, not a fact")
3. Increase M(1).V — trust sensory evidence again ("pay attention to what actually happens when you try")
4. Increase M(1).pC — allow templates to update ("what you think moderation feels like may be outdated")

The agent does the rest. Active inference drives it toward the attractor that best fits reality — once the precision balance allows evidence to flow.
