function dx = psychology_state_model_f(x, v, P)
% Generative model dynamics for reserves, accumulated fatigue, and effort.

if isempty(P)
    [P, ~, ~, ~] = psychology_defaults();
end

[state, ~] = psychology_state_unpack(x, P);

targetEffort = P.driveOffset ...
    + P.driveEnergy  * v(1) ...
    + P.driveReward  * v(2) ...
    - P.driveFatigue * v(3) ...
    + P.driveReserve * state.reserves ...
    - P.driveStateFatigue * state.fatigueState;

dx         = zeros(3, 1);
dx(1)      = P.reserveRecover * (P.reserveSetpoint - state.reserves) ...
    - P.reserveSpend * state.energy;
dx(2)      = P.fatigueBuild * state.energy.^2 ...
    - P.fatigueDecay * state.fatigueState;
dx(3)      = (targetEffort - x(3)) / P.effortTau;

end
