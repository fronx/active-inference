import { Expr, field } from './expr.js';

export interface BeliefSpec {
  shape: number[];
  init: 'random' | 'zeros';
}

export interface ActionSpec {
  shape: number[];
  init: 'random' | 'zeros';
}

export interface PriorSpec {
  shape: number[];
  init: 'random' | 'zeros';
}

export interface ModelConfig {
  timesteps: number;
  beliefs?: Record<string, BeliefSpec>;
  actions?: Record<string, ActionSpec>;
  priors?: Record<string, PriorSpec>;
  observations?: string[];
}

type FieldProxy = Expr & Record<string, Expr>;
type GenerativeModelFn = (v: FieldProxy, P: FieldProxy) => Record<string, Expr>;

export class ActiveInferenceModel {
  public readonly timesteps: number;
  public readonly beliefs: Record<string, BeliefSpec>;
  public readonly actions: Record<string, ActionSpec>;
  public readonly priors: Record<string, PriorSpec>;
  public readonly observations: string[];
  public generativeModelCode: string = '';
  public generativeProcessCode: string = '';

  constructor(config: ModelConfig) {
    this.timesteps = config.timesteps;
    this.beliefs = config.beliefs ?? {};
    this.actions = config.actions ?? {};
    this.priors = config.priors ?? {};
    this.observations = config.observations ?? [];
  }

  setGenerativeModel(code: string): void {
    this.generativeModelCode = code;
  }

  setGenerativeModelFn(fn: GenerativeModelFn): void {
    const v = field('v');
    const P = field('P');
    const outputs = fn(v, P);

    const lines: string[] = [];
    const intermediates = new Map<string, string>();

    // Extract intermediate variables
    for (const [name, expr] of Object.entries(outputs)) {
      const matlabCode = expr.toMATLAB();

      // Check if expression contains intermediate computation
      const softmaxMatch = matlabCode.match(/spm_softmax\(v\)/);
      if (softmaxMatch && !intermediates.has('identity')) {
        intermediates.set('identity', 'spm_softmax(v)');
        lines.push('  identity = spm_softmax(v);');
      }
    }

    // Generate output assignments
    for (const [name, expr] of Object.entries(outputs)) {
      let matlabCode = expr.toMATLAB();
      // Replace intermediate values
      if (intermediates.has('identity')) {
        matlabCode = matlabCode.replace(/spm_softmax\(v\)/g, 'identity');
      }
      lines.push(`  g.${name} = ${matlabCode};`);
    }

    this.generativeModelCode = lines.join('\n');
  }

  setGenerativeProcess(code: string): void {
    this.generativeProcessCode = code;
  }

  generateMATLABFile(): string {
    const sections: string[] = [];

    // Helper functions
    sections.push(`function g = expect(x, v, P)`);
    sections.push(this.generativeModelCode);
    sections.push(`end\n`);

    sections.push(`function g = observe(x, v, action, P)`);
    sections.push(this.generativeProcessCode);
    sections.push(`end\n`);

    // Main script
    sections.push(`% Initialization`);
    sections.push(this.generateInitCode());

    // Prior parameters (P structure)
    if (Object.keys(this.priors).length > 0) {
      sections.push(``);
      sections.push(`% Prior parameters (target configuration)`);
      for (const [name, spec] of Object.entries(this.priors)) {
        const initFn = spec.init === 'random' ? 'randn' : 'zeros';
        sections.push(`P.${name} = ${initFn}(${spec.shape.join(', ')});`);
      }
    }

    sections.push(``);

    // Build DEM structure directly
    sections.push(`% DEM structure`);
    sections.push(`DEM.M(1).g = @expect;`);
    if (Object.keys(this.priors).length > 0) {
      sections.push(`DEM.M(1).pE = P;`);
    }
    sections.push(`DEM.G(1).g = @observe;`);
    sections.push(`DEM.C = zeros(1, ${this.timesteps});`);
    sections.push(``);

    // Run solver
    sections.push(`% Run active inference`);
    sections.push(`DEM = spm_ADEM(DEM);`);

    return sections.join('\n');
  }

  generateInitCode(): string {
    const lines: string[] = [];

    const addInitLines = (specs: Record<string, BeliefSpec | ActionSpec>) => {
      for (const [name, spec] of Object.entries(specs)) {
        const initFn = spec.init === 'random' ? 'randn' : 'zeros';
        lines.push(`${name} = ${initFn}(${spec.shape.join(', ')})`);
      }
    };

    addInitLines(this.beliefs);
    addInitLines(this.actions);

    return lines.join(';\n');
  }
}
