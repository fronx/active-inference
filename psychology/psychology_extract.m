function traces = psychology_extract(DEM, N)
% Extract time series from a solved DEM structure.
%
% Returns a struct with belief, observation, and hidden-state trajectories.

[P, ~, ~, ~, action, observation] = psychology_defaults();
if isfield(DEM, 'PG')
    P = DEM.PG;
end

traces = struct();
traces.beliefs  = zeros(3, N);
traces.energy   = zeros(1, N);
traces.feltEnergy = zeros(1, N);
traces.reward   = zeros(1, N);
traces.fatigue  = zeros(1, N);
traces.reserves = zeros(1, N);
traces.activation = zeros(1, N);
traces.effort     = zeros(1, N);
traces.fatigueState = zeros(1, N);
traces.capacity     = zeros(1, N);
traces.actionTarget = zeros(1, N);
traces.opportunity = zeros(1, N);

for t = 1:N
    traces.beliefs(:, t) = DEM.qU.v{2}(:, t);

    a = spm_unvec(DEM.qU.a{2}(:, t), action);
    y = spm_unvec(DEM.pU.v{1}(:, t), observation);
    [state0, ~] = psychology_state_unpack(DEM.pU.x{1}(:, t), P, 0, t / N);
    bodyLeverage = min(state0.reserves / max(state0.capacity, exp(-8)), 1);
    [state, ~] = psychology_state_unpack(DEM.pU.x{1}(:, t), P, ...
        P.actionActivationGain * a.energy * bodyLeverage, t / N);

    traces.actionTarget(t) = a.energy;
    traces.energy(t)       = y.energy;
    traces.feltEnergy(t)   = y.feltEnergy;
    traces.reward(t)       = y.reward;
    traces.fatigue(t)      = y.fatigue;
    traces.reserves(t)     = state.reserves;
    traces.activation(t)   = state.activation;
    traces.effort(t)       = state.activation;
    traces.fatigueState(t) = state.fatigueState;
    traces.capacity(t)     = state.capacity;
    traces.opportunity(t)  = state.opportunity;
end

end
