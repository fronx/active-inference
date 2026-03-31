function traces = psychology_extract(DEM, N)
% Extract time series from a solved DEM structure.
%
% Returns a struct with belief, observation, and hidden-state trajectories.

[P, ~, action, observation] = psychology_defaults();
if isfield(DEM, 'P')
    P = DEM.P;
end

traces = struct();
traces.beliefs  = zeros(3, N);
traces.energy   = zeros(1, N);
traces.reward   = zeros(1, N);
traces.fatigue  = zeros(1, N);
traces.reserves = zeros(1, N);
traces.effort   = zeros(1, N);
traces.fatigueState = zeros(1, N);
traces.actionTarget = zeros(1, N);

for t = 1:N
    traces.beliefs(:, t) = DEM.qU.v{2}(:, t);

    a = spm_unvec(DEM.qU.a{2}(:, t), action);
    y = spm_unvec(DEM.pU.v{1}(:, t), observation);
    [state, ~] = psychology_state_unpack(DEM.pU.x{1}(:, t), P, ...
        P.actionEffortGain * a.energy);

    traces.actionTarget(t) = a.energy;
    traces.energy(t)       = y.energy;
    traces.reward(t)       = y.reward;
    traces.fatigue(t)      = y.fatigue;
    traces.reserves(t)     = state.reserves;
    traces.effort(t)       = state.effort;
    traces.fatigueState(t) = state.fatigueState;
end

end
