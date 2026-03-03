import { ActiveInferenceModel } from './model.js';

// Example: Using TypeScript DSL for type-safe model definition
const model = new ActiveInferenceModel({
  timesteps: 32,
  beliefs: {
    identityLogits: { shape: [16, 16], init: 'random' },
  },
  actions: {
    position: { shape: [2, 16], init: 'zeros' },
    signal: { shape: [3, 16], init: 'zeros' },
  },
  priors: {
    position: { shape: [2, 16], init: 'zeros' },
    signal: { shape: [3, 16], init: 'zeros' },
    sense: { shape: [3, 16], init: 'zeros' },
  },
  observations: ['position', 'signal', 'sense'],
});

// Define generative model using TypeScript DSL
// Type-safe with IDE autocomplete!
model.setGenerativeModelFn((v, P) => {
  const identity = v.softmax();
  return {
    position: P.position.multiply(identity),
    signal: P.signal.multiply(identity),
    sense: P.sense.multiply(identity),
  };
});

// Still support raw MATLAB for complex cases
model.setGenerativeProcess(`
  g.position = action.position;
  g.signal = action.signal;
  g.sense = morphogenesis(action.position, action.signal);
`);

// Generate complete MATLAB file
const matlabCode = model.generateMATLABFile();

console.log('Generated MATLAB code with TypeScript DSL:');
console.log('='.repeat(60));
console.log(matlabCode);
console.log('='.repeat(60));
