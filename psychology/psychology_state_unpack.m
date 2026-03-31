function [state, observation] = psychology_state_unpack(x, P, actionShift, tau)
% Convert latent states into interpretable quantities and observations.

if isempty(P)
    [P, ~, ~, ~, ~, ~] = psychology_defaults();
end

if isempty(x)
    x = zeros(4, 1);
end

if nargin < 3
    actionShift = 0;
end

if nargin < 4
    tau = [];
end

opportunity = psychology_opportunity(P, tau);

state = struct();
state.reserves      = psychology_softplus(x(1), P.softplusScale);
state.fatigueState  = psychology_softplus(x(2), P.softplusScale);
state.activationRaw = P.activationMax * psychology_sigmoid(x(3));
state.activation    = P.activationMax * psychology_sigmoid(x(3) + actionShift);
state.capacity      = psychology_softplus(x(4), P.softplusScale);
state.energy        = state.reserves * state.activation;
state.opportunity   = opportunity;
state.bodyAvailability = min(state.reserves / max(state.capacity, exp(-8)), 1);

observation = struct();
observation.energy      = state.energy;
observation.feltEnergy  = max( ...
    (P.feltEnergyBodyFloor + (1 - P.feltEnergyBodyFloor) * state.bodyAvailability) ...
    * state.activation ...
    + P.feltEnergyOpportunityBoost * opportunity ...
    - P.feltEnergyFatiguePenalty * state.fatigueState, ...
    0);
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
