function bias = psychology_regime_bias(beliefPrior)
% Continuous regime score derived from the existing belief priors.
% Negative = depressed-like, positive = manic-like.

bias = 0.80 * (beliefPrior(1) - 1.0) ...
    + 1.10 * (beliefPrior(2) - 0.5) ...
    - 0.90 * (beliefPrior(3) - 0.3);

end
