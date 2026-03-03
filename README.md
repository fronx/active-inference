# Active Inference Morphogenesis Simulation

Reproduces the simulations from [Friston et al. 2015 - "Knowing one's place: a free-energy approach to pattern regulation"](https://pmc.ncbi.nlm.nih.gov/articles/PMC4387527/).

## Requirements

- macOS with Homebrew
- GNU Octave: `brew install octave`

## Run

```bash
octave --gui run_morphogenesis.m
```

### Psychology Toy Model

Generate a minimal affective dynamics example that contrasts depressive (down-regulated) and manic (up-regulated) energy biases:

```bash
npm run example:psychology > dem_psychology.m
octave --gui dem_psychology.m
```

The emitted MATLAB/Octave file defines a single-degree-of-freedom controller (`energyPolicy`) with biased feedback loops so you can explore how dysregulated energy expenditure shapes its own trajectory.

## What it does

16 cells start at identical positions and self-assemble into a target morphology (head-body-tail) over 32 time steps using active inference (free energy minimization).

## Documentation

- [Architecture](docs/architecture.md) - Detailed flow diagram showing how the SPM12 library and custom model code interact
- [Psychology Toy Model](docs/psychology-toy-model.md) - Rationale and math for the manic/depressive energy bias example

## Notes

- `spm12/spm_platform.m` was patched to support Apple Silicon (arm64)
- SPM12 source: https://github.com/spm/spm12
