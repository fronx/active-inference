import { ActiveInferenceModel } from './model.js';

// Example: Simple morphogenesis model
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

// Define generative model (M): how beliefs predict observations
model.setGenerativeModel(`
  % Convert identity logits to probabilities
  identity = spm_softmax(v);

  % Expected observations given identity beliefs
  g.position = P.position * identity;
  g.signal = P.signal * identity;
  g.sense = P.sense * identity;
`);

// Define generative process (G): how actions generate observations
model.setGenerativeProcess(`
  % Observations are directly determined by actions
  g.position = action.position;
  g.signal = action.signal;

  % Sense is computed via morphogenesis dynamics
  g.sense = morphogenesis(action.position, action.signal);
`);

// Generate complete MATLAB file
const matlabCode = model.generateMATLABFile();

console.log('Generated MATLAB code:');
console.log('='.repeat(60));
console.log(matlabCode);
console.log('='.repeat(60));
