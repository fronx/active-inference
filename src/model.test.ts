import { describe, it, expect } from 'vitest';
import { ActiveInferenceModel } from './model.js';

describe('ActiveInferenceModel', () => {
  it('creates a model with timesteps', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
    });

    expect(model.timesteps).toBe(32);
  });

  it('defines beliefs with initial values', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      beliefs: {
        identityLogits: { shape: [16, 16], init: 'random' },
      },
    });

    expect(model.beliefs).toBeDefined();
    expect(model.beliefs.identityLogits).toEqual({
      shape: [16, 16],
      init: 'random',
    });
  });

  it('defines actions with initial values', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      actions: {
        position: { shape: [2, 16], init: 'zeros' },
        signal: { shape: [3, 16], init: 'zeros' },
      },
    });

    expect(model.actions).toBeDefined();
    expect(model.actions.position).toEqual({ shape: [2, 16], init: 'zeros' });
    expect(model.actions.signal).toEqual({ shape: [3, 16], init: 'zeros' });
  });

  it('generates MATLAB initialization code for random beliefs', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      beliefs: {
        identityLogits: { shape: [16, 16], init: 'random' },
      },
    });

    const matlabCode = model.generateInitCode();

    expect(matlabCode).toContain('identityLogits = randn(16, 16)');
  });

  it('generates MATLAB initialization code for actions with zeros', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      actions: {
        position: { shape: [2, 16], init: 'zeros' },
        signal: { shape: [3, 16], init: 'zeros' },
      },
    });

    const matlabCode = model.generateInitCode();

    expect(matlabCode).toContain('position = zeros(2, 16)');
    expect(matlabCode).toContain('signal = zeros(3, 16)');
  });

  it('defines observable fields', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      observations: ['position', 'signal', 'sense'],
    });

    expect(model.observations).toEqual(['position', 'signal', 'sense']);
  });

  it('defines generative model function with MATLAB code', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
    });

    model.setGenerativeModel(`
      identity = spm_softmax(v);
      g.position = P.position * identity;
      g.signal = P.signal * identity;
    `);

    expect(model.generativeModelCode).toContain('spm_softmax');
    expect(model.generativeModelCode).toContain('g.position');
  });

  it('defines generative process function with MATLAB code', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
    });

    model.setGenerativeProcess(`
      g.position = action.position;
      g.signal = action.signal;
      g.sense = morphogenesis(action.position, action.signal);
    `);

    expect(model.generativeProcessCode).toContain('action.position');
    expect(model.generativeProcessCode).toContain('morphogenesis');
  });

  it('generates complete MATLAB file with DEM structure', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      beliefs: {
        identityLogits: { shape: [16, 16], init: 'random' },
      },
      actions: {
        position: { shape: [2, 16], init: 'zeros' },
      },
    });

    model.setGenerativeModel(`
      identity = spm_softmax(v);
      g.position = P.position * identity;
    `);

    model.setGenerativeProcess(`
      g.position = action.position;
    `);

    const matlabFile = model.generateMATLABFile();

    expect(matlabFile).toContain('function g = expect(x, v, P)');
    expect(matlabFile).toContain('function g = observe(x, v, action, P)');
    expect(matlabFile).toContain('DEM.M(1).g = @expect');
    expect(matlabFile).toContain('DEM.G(1).g = @observe');
    expect(matlabFile).toContain('DEM = spm_ADEM(DEM)');
  });

  it('defines prior parameters for target configuration', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      priors: {
        position: { shape: [2, 16], init: 'zeros' },
        signal: { shape: [3, 16], init: 'zeros' },
      },
    });

    expect(model.priors).toBeDefined();
    expect(model.priors.position).toEqual({ shape: [2, 16], init: 'zeros' });
  });

  it('generates MATLAB code with prior initialization and P structure', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
      priors: {
        position: { shape: [2, 16], init: 'zeros' },
        signal: { shape: [3, 16], init: 'zeros' },
      },
    });

    const matlabCode = model.generateMATLABFile();

    expect(matlabCode).toContain('P.position = zeros(2, 16)');
    expect(matlabCode).toContain('P.signal = zeros(3, 16)');
    expect(matlabCode).toContain('DEM.M(1).pE = P');
  });

  it('defines generative model with TypeScript expression builder', () => {
    const model = new ActiveInferenceModel({
      timesteps: 32,
    });

    model.setGenerativeModelFn((v, P) => {
      const identity = v.softmax();
      return {
        position: P.position.multiply(identity),
        signal: P.signal.multiply(identity),
      };
    });

    const matlabCode = model.generateMATLABFile();

    expect(matlabCode).toContain('identity = spm_softmax(v)');
    expect(matlabCode).toContain('g.position = P.position * identity');
    expect(matlabCode).toContain('g.signal = P.signal * identity');
  });
});
