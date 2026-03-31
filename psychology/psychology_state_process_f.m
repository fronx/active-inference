function dx = psychology_state_process_f(x, v, a, P)
% Generative process dynamics driven by opportunity and action.

if isempty(P)
    [P, ~, ~, ~, action, ~] = psychology_defaults();
else
    [~, ~, ~, ~, action, ~] = psychology_defaults();
end

if isempty(a)
    a = action;
end

a = spm_unvec(a, action);
[state0, ~] = psychology_state_unpack(x, P, 0);
actionShift = action_shift(P, state0, a.energy);
[state, ~] = psychology_state_unpack(x, P, actionShift);

targetActivation = P.activationBaseline ...
    + P.activationOpportunity * state.opportunity ...
    - P.activationFatigueDrag * state.fatigueState ...
    - P.activationReserveDrag * smooth_positive(state.capacity - state.reserves, P.softplusScale);

rest = max(1 - state.activation / P.activationMax, 0);
moderate = state.opportunity ...
    * exp(-((state.energy - P.moderateUseCenter).^2) / (2 * P.moderateUseWidth^2));
underuse = state.opportunity ...
    * psychology_sigmoid((P.underuseThreshold - state.energy) / P.underuseScale);
overuse  = state.opportunity ...
    * psychology_sigmoid((state.energy - P.overuseThreshold) / P.overuseScale);

dx    = zeros(4, 1);
dx(1) = P.reserveRecover * (state.capacity - state.reserves) * rest ...
    - P.reserveSpend * state.energy;
dx(2) = P.fatigueBuild * state.energy ...
    - P.fatigueDecay * rest * state.fatigueState;
dx(3) = (targetActivation - x(3)) / P.activationTau;
dx(4) = P.capacityHomeostasis * (P.capacityBaseline - state.capacity) ...
    + P.capacityBuild * moderate ...
    - P.capacityAtrophy * underuse ...
    - P.capacityDamage * overuse;

end

function y = smooth_positive(x, scale)
y = log1p(exp(scale * x)) / scale;
end

function y = psychology_sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function shift = action_shift(P, state, actionEnergy)
bodyLeverage = min(state.reserves / max(state.capacity, exp(-8)), 1);
shift = P.actionActivationGain * actionEnergy * bodyLeverage;
end
