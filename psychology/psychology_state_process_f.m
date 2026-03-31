function dx = psychology_state_process_f(x, v, a, P)
% Generative process dynamics driven by action.

if isempty(P)
    [P, ~, action, ~] = psychology_defaults();
else
    [~, ~, action, ~] = psychology_defaults();
end

if isempty(a)
    a = action;
end

a = spm_unvec(a, action);
[state, ~] = psychology_state_unpack(x, P, P.actionEffortGain * a.energy);

dx         = zeros(3, 1);
dx(1)      = P.reserveRecover * (P.reserveSetpoint - state.reserves) ...
    - P.reserveSpend * state.energy;
dx(2)      = P.fatigueBuild * state.energy.^2 ...
    - P.fatigueDecay * state.fatigueState;
dx(3)      = (P.tonicEffortBaseline - x(3)) / P.effortTau;

end
