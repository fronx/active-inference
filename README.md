# Active Inference Morphogenesis Simulation

Reproduces the simulations from [Friston et al. 2015 - "Knowing one's place: a free-energy approach to pattern regulation"](https://pmc.ncbi.nlm.nih.gov/articles/PMC4387527/).

## Requirements

- macOS with Homebrew
- GNU Octave: `brew install octave`

## Run

```bash
octave --gui run_morphogenesis.m
```

## What it does

16 cells start at identical positions and self-assemble into a target morphology (head-body-tail) over 32 time steps using active inference (free energy minimization).

## Documentation

- [Architecture](docs/architecture.md) - Detailed flow diagram showing how the SPM12 library and custom model code interact

## Notes

- `spm12/spm_platform.m` was patched to support Apple Silicon (arm64)
- SPM12 source: https://github.com/spm/spm12
