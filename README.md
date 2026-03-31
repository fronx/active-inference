# Active Inference

Exploring active inference by building toy models on SPM12's `spm_ADEM` solver.

## Requirements

- macOS with Homebrew
- GNU Octave: `brew install octave`

## Models

### Morphogenesis

Reproduces [Friston et al. 2015 - "Knowing one's place: a free-energy approach to pattern regulation"](https://pmc.ncbi.nlm.nih.gov/articles/PMC4387527/). 16 cells start at identical positions and self-assemble into a target morphology (head-body-tail) over 32 time steps using free energy minimization.

```bash
octave --gui run_morphogenesis.m
```

### Psychology

An agent expends energy and observes what comes back: reward (diminishing returns) and fatigue (quadratic cost). It maintains core beliefs about self-efficacy — "effort pays off," "moderate returns," or "effort is futile" — that shape how much energy it expends, which shapes what it observes, which reinforces the beliefs.

Same generative process, same model structure, different precision balance:

- **Healthy**: loose prior, high sensory precision — beliefs update from evidence, agent converges to moderate effort
- **Depressed**: rigid prior toward "effort is futile," low sensory precision — low effort produces low reward, confirming the belief
- **Manic**: rigid prior toward "effort always pays off," low sensory precision — high effort hits diminishing returns, but prediction errors are ignored

```bash
octave --gui run_psychology.m
```

See [psychology.md](psychology.md) for the full model description.

## Web UI

Interactive browser interface for the psychology model. Lets you pick presets (Healthy, Depressed, Manic) or tweak parameters with sliders, and streams simulation results as real-time charts.

Requires Node.js and GNU Octave.

```bash
# Terminal 1: start the server (runs Octave in the background)
cd web/server
npm install
npm run dev

# Terminal 2: start the client
cd web/client
npm install
npm run dev
```

Then open http://localhost:5173.

## Documentation

- [Architecture](docs/architecture.md) - How the SPM12 library and custom model code interact

## Notes

- `spm12/spm_platform.m` was patched to support Apple Silicon (arm64)
- SPM12 source: https://github.com/spm/spm12
