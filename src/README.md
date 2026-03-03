# TypeScript Wrapper for SPM Active Inference

A TypeScript wrapper that provides a readable API for defining active inference models and generating Octave/MATLAB code for SPM12.

## Quick Start

```typescript
import { ActiveInferenceModel } from './model.js';

// Create a model
const model = new ActiveInferenceModel({
  timesteps: 32,
  beliefs: {
    identityLogits: { shape: [16, 16], init: 'random' },
  },
  actions: {
    position: { shape: [2, 16], init: 'zeros' },
    signal: { shape: [3, 16], init: 'zeros' },
  },
  observations: ['position', 'signal', 'sense'],
});

// Define how beliefs predict observations (generative model)
model.setGenerativeModel(`
  identity = spm_softmax(v);
  g.position = P.position * identity;
  g.signal = P.signal * identity;
`);

// Define how actions generate observations (generative process)
model.setGenerativeProcess(`
  g.position = action.position;
  g.signal = action.signal;
  g.sense = morphogenesis(action.position, action.signal);
`);

// Generate MATLAB/Octave code
const matlabCode = model.generateMATLABFile();
console.log(matlabCode);
```

## Running the Example

```bash
npm run example
```

## Running Tests

```bash
npm test
```

## API

### `ActiveInferenceModel(config)`

Creates a new model.

**Config:**
- `timesteps` (number): Number of time steps for the simulation
- `beliefs` (optional): Record of belief variables with shape and initialization
- `actions` (optional): Record of action variables with shape and initialization
- `observations` (optional): Array of observable field names

### `setGenerativeModel(code: string)`

Defines the generative model function (`expect()` in MATLAB). This maps beliefs to predicted observations.

**Parameters in the function:**
- `x` - hidden states (unused in morphogenesis)
- `v` - hidden causes (belief logits)
- `P` - prior parameters (target configuration)

**Returns:** `g` - expected observations

### `setGenerativeProcess(code: string)`

Defines the generative process function (`observe()` in MATLAB). This maps actions to actual observations.

**Parameters in the function:**
- `x` - hidden states (unused)
- `v` - hidden causes (unused)
- `action` - action parameters
- `P` - prior parameters

**Returns:** `g` - actual observations

### `generateInitCode(): string`

Generates MATLAB initialization code for beliefs and actions.

### `generateMATLABFile(): string`

Generates a complete MATLAB/Octave file with:
- Function definitions (`expect`, `observe`)
- Variable initialization
- DEM structure setup
- `spm_ADEM` call

## How It Maps to SPM

The wrapper generates code that follows the SPM12 active inference interface:

```
TypeScript                     MATLAB/SPM
──────────────────────────────────────────────
beliefs                    →   M(2).v (hidden causes)
actions                    →   G(2).a (action variables)
setGenerativeModel()       →   M(1).g = @expect
setGenerativeProcess()     →   G(1).g = @observe
timesteps                  →   DEM.C (length)
```

## Development

Built with TDD using Vitest. All features have comprehensive test coverage.

```bash
npm test              # Run tests
npm run test:ui       # Run tests with UI
npm run build         # Compile TypeScript
```
