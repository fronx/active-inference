function dx = psychology_state_process_f(x, v, a, P)
% Generative process dynamics driven by opportunity and action.

if isempty(P)
    [P, ~, action, ~] = psychology_defaults();
else
    [~, ~, action, ~] = psychology_defaults();
end

if isempty(a)
    a = action;
end

a = spm_unvec(a, action);
[state, ~] = psychology_state_unpack(x, P, P.actionActivationGain * a.energy);

targetActivation = P.activationBaseline ...
    + P.activationOpportunity * state.opportunity ...
    - P.activationFatigueDrag * state.fatigueState ...
    - P.activationReserveDrag * max(P.reserveSetpoint - state.reserves, 0);

rest = max(1 - state.activation / P.activationMax, 0);

dx    = zeros(3, 1);
dx(1) = P.reserveRecover * (P.reserveSetpoint - state.reserves) * rest ...
    - P.reserveSpend * state.energy;
dx(2) = P.fatigueBuild * state.energy ...
    - P.fatigueDecay * rest * state.fatigueState;
dx(3) = (targetActivation - x(3)) / P.activationTau;

end
