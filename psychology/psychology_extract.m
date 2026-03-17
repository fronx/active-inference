function [beliefs, energies, rewards, fatigues] = psychology_extract(DEM, N, action)
% Extract time series from a solved DEM structure.
%
% Returns:
%   beliefs  - 3 x N matrix: [energy; reward; fatigue] expectations over time
%   energies - 1 x N vector of energy expenditure (action)
%   rewards  - 1 x N vector of actual reward (from generative process)
%   fatigues - 1 x N vector of actual fatigue (from generative process)

beliefs  = zeros(3, N);
energies = zeros(1, N);
rewards  = zeros(1, N);
fatigues = zeros(1, N);

for t = 1:N
    beliefs(:, t) = DEM.qU.v{2}(:, t);

    a = spm_unvec(DEM.qU.a{2}(:, t), action);
    e = a.energy;
    energies(t) = e;
    rewards(t)  = e / (1 + abs(e));
    fatigues(t) = 0.3 * e^2;
end

end
