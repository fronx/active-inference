function [state, observation] = psychology_state_unpack(x, P, effortShift)
% Convert latent states into interpretable quantities and observations.

if isempty(P)
    [P, ~, ~, ~] = psychology_defaults();
end

if isempty(x)
    x = zeros(3, 1);
end

if nargin < 3
    effortShift = 0;
end

state = struct();
state.reserves     = P.reserveMax * psychology_sigmoid(x(1));
state.fatigueState = psychology_softplus(x(2), P.softplusScale);
state.effort       = P.effortMax * psychology_sigmoid(x(3) + effortShift);

state.energy = state.reserves * state.effort;

observation = struct();
observation.energy  = state.energy;
observation.reward  = state.energy / (1 + state.energy) ...
    - P.rewardFatiguePenalty * state.fatigueState;
observation.fatigue = state.fatigueState + P.observedFatigueLoad * state.energy.^2;

end

function y = psychology_sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function y = psychology_softplus(x, scale)
y = log1p(exp(scale * x)) / scale;
end
