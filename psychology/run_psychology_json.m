function run_psychology_json(params_json)
% Run psychology simulation and output results as JSON to stdout.
%
% Input: JSON string with fields:
%   beliefPrior - [3] array: [expected_energy, expected_reward, expected_fatigue]
%   M2V, M1V   - precision exponents (scalars)
%   N           - time steps (scalar)

params = jsondecode(params_json);

beliefPrior = params.beliefPrior(:);
N           = params.N;
M2V         = params.M2V;
M1V         = params.M1V;

DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);
traces = psychology_extract(DEM, N);

result.timesteps        = 1:N;
result.freeEnergy       = -DEM.J;
result.beliefs.energy   = traces.beliefs(1, :);
result.beliefs.reward   = traces.beliefs(2, :);
result.beliefs.fatigue  = traces.beliefs(3, :);
result.energy           = traces.energy;
result.reward           = traces.reward;
result.fatigue          = traces.fatigue;
result.reserves         = traces.reserves;
result.effort           = traces.effort;
result.fatigueState     = traces.fatigueState;
result.actionTarget     = traces.actionTarget;

fprintf('%s\n', jsonencode(result));

end
