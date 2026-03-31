function dx = psychology_state_model_f(x, v, P)
% Generative model dynamics for reserves, fatigue, and activation.

if isempty(P)
    [P, ~, ~, ~] = psychology_defaults();
end

[state, ~] = psychology_state_unpack(x, P, 0);

targetActivation = P.activationBaseline ...
    + P.activationOpportunity * state.opportunity ...
    + P.modelDriveEnergy  * v(1) ...
    + P.modelDriveReward  * v(2) ...
    - P.modelDriveFatigue * v(3) ...
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
