import { describe, it, expect } from 'vitest';
import { softmax, field } from './expr.js';

describe('Expression Builder', () => {
  it('creates softmax expression', () => {
    const expr = softmax('v');

    expect(expr.toMATLAB()).toBe('spm_softmax(v)');
  });

  it('creates multiply expression', () => {
    const expr = softmax('v');
    const result = expr.multiply('P.position');

    expect(result.toMATLAB()).toBe('spm_softmax(v) * P.position');
  });

  it('creates field access expression', () => {
    const P = field('P');
    const position = P.position;

    expect(position.toMATLAB()).toBe('P.position');
  });

  it('chains field access with operations', () => {
    const P = field('P');
    const v = field('v');
    const identity = softmax(v);
    const result = P.position.multiply(identity);

    expect(result.toMATLAB()).toBe('P.position * spm_softmax(v)');
  });
});
