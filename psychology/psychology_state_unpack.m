function [state, observation] = psychology_state_unpack(x, P, actionShift, tau)
% Convert latent states into interpretable quantities and observations.

if isempty(P)
    [P, ~, ~, ~] = psychology_defaults();
end

if isempty(x)
    x = zeros(3, 1);
end

if nargin < 3
    actionShift = 0;
end

if nargin < 4
    tau = [];
end

opportunity = psychology_opportunity(P, tau);

state = struct();
state.reserves      = P.reserveMax * psychology_sigmoid(x(1));
state.fatigueState  = psychology_softplus(x(2), P.softplusScale);
state.activationRaw = P.activationMax * psychology_sigmoid(x(3));
state.activation    = P.activationMax * psychology_sigmoid(x(3) + actionShift);
state.energy        = state.reserves * state.activation;
state.opportunity   = opportunity;

observation = struct();
observation.energy      = state.energy;
observation.reward      = opportunity * (state.energy / (1 + state.energy)) ...
    - P.rewardFatiguePenalty * state.fatigueState;
observation.fatigue     = state.fatigueState + P.observedFatigueLoad * state.energy.^2;
observation.opportunity = opportunity;

end

function y = psychology_sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function y = psychology_softplus(x, scale)
y = log1p(exp(scale * x)) / scale;
end
