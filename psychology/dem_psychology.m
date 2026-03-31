function DEM = dem_psychology(regime)
% Active inference model of energy expenditure dysregulation.
%
% Demonstrates how depression and mania emerge as precision pathologies:
% same generative process, same model structure, different precision balance.
%
% Usage:
%   DEM = dem_psychology('healthy')
%   DEM = dem_psychology('depressed')
%   DEM = dem_psychology('manic')

if nargin < 1; regime = 'healthy'; end

% parameters
%--------------------------------------------------------------------------
[N, beliefPrior, M2V, M1V] = psychology_params(regime);

% solve
%==========================================================================
DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);

% extract time series for plotting
%--------------------------------------------------------------------------
traces = psychology_extract(DEM, N);

% Graphics
%==========================================================================
spm_figure('GetWin', ['Psychology: ' regime]); clf

subplot(2, 2, 1)
plot(-DEM.J)
title(sprintf('Free energy (%s)', regime), 'FontSize', 14)
xlabel('time')
ylabel('free energy')
axis square tight
grid on

subplot(2, 2, 2)
plot(1:N, traces.beliefs')
legend({'expected vitality', 'expected reward', 'expected fatigue'}, 'Location', 'best')
title('Beliefs about the world', 'FontSize', 14)
xlabel('time')
ylabel('expectation')
axis square tight
grid on

subplot(2, 2, 3)
plot(1:N, traces.energy, 'LineWidth', 2)
hold on
plot(1:N, traces.feltEnergy, '-', 'Color', [0.65 0.08 0.95], 'LineWidth', 1.5)
plot(1:N, traces.reserves, '--b', 'LineWidth', 1.5)
plot(1:N, traces.activation, ':m', 'LineWidth', 1.5)
plot(1:N, traces.opportunity, '-.k', 'LineWidth', 1.2)
legend({'realized energy', 'felt energy', 'reserves', 'activation', 'opportunity'}, 'Location', 'best')
title('Energy, Vigor, and Opportunity', 'FontSize', 14)
xlabel('time')
ylabel('level')
axis square tight
grid on

subplot(2, 2, 4)
plot(1:N, traces.reward, 'g', 'LineWidth', 2); hold on
plot(1:N, traces.fatigue, 'r', 'LineWidth', 2)
plot(1:N, traces.fatigueState, '--k', 'LineWidth', 1.5)
legend({'reward', 'observed fatigue', 'fatigue state'}, 'Location', 'best')
title('Reward and Recovery Cost', 'FontSize', 14)
xlabel('time')
ylabel('level')
axis square tight
grid on

end
