export interface Expr {
  toMATLAB(): string;
  multiply(other: string | Expr): Expr;
  softmax(): Expr;
}

function createExpr(matlabCode: string): Expr {
  return {
    toMATLAB() {
      return matlabCode;
    },
    multiply(other: string | Expr): Expr {
      const otherStr = typeof other === 'string' ? other : other.toMATLAB();
      return createExpr(`${matlabCode} * ${otherStr}`);
    },
    softmax(): Expr {
      return createExpr(`spm_softmax(${matlabCode})`);
    },
  };
}

export function softmax(arg: string | Expr): Expr {
  const argStr = typeof arg === 'string' ? arg : arg.toMATLAB();
  return createExpr(`spm_softmax(${argStr})`);
}

export function field(name: string): Expr & Record<string, Expr> {
  return new Proxy(createExpr(name), {
    get(target, prop) {
      if (prop === 'toMATLAB' || prop === 'multiply' || prop === 'softmax') {
        return target[prop as keyof Expr];
      }
      return field(`${name}.${String(prop)}`);
    },
  }) as Expr & Record<string, Expr>;
}
