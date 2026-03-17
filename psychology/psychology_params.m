function [N, beliefPrior, M2V, M1V] = psychology_params(regime)
% Return default parameters for a named regime.
%
% beliefPrior = [expected_energy; expected_reward; expected_fatigue]
% These are direct beliefs about the world, not categorical self-models.
%
% At energy=1.0, reality gives reward=0.5, fatigue=0.3.
% Healthy priors match reality. Pathological priors diverge from it.

if nargin < 1; regime = 'healthy'; end

N = 48;

switch regime
    case 'healthy'
        beliefPrior = [1.0; 0.5; 0.3];    % realistic expectations
        M2V        = -2;                   % loose prior (updates from evidence)
        M1V        =  3;                   % trusts sensory evidence

    case 'depressed'
        beliefPrior = [0.6; 0.2; 0.8];    % "not worth my energy"
        M2V        =  4;                   % rigid prior (resists evidence)
        M1V        =  0;                   % discounts sensory evidence

    case 'manic'
        beliefPrior = [1.4; 0.8; 0.2];    % "I can do anything"
        M2V        =  4;                   % rigid prior
        M1V        =  0;                   % discounts sensory evidence

    otherwise
        error('Unknown regime: %s', regime);
end

end
