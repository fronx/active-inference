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
action.energy = 0;
[beliefs, energies, rewards, fatigues] = psychology_extract(DEM, N, action);

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
plot(1:N, beliefs')
legend({'expected energy', 'expected reward', 'expected fatigue'}, 'Location', 'best')
title('Beliefs about the world', 'FontSize', 14)
xlabel('time')
ylabel('expectation')
axis square tight
grid on

subplot(2, 2, 3)
plot(1:N, energies, 'LineWidth', 2)
hold on
plot([1 N], [1.0 1.0], '--k')
title('Energy expenditure', 'FontSize', 14)
xlabel('time')
ylabel('energy')
axis square tight
grid on

subplot(2, 2, 4)
plot(1:N, rewards, 'g', 'LineWidth', 2); hold on
plot(1:N, fatigues, 'r', 'LineWidth', 2)
legend({'reward', 'fatigue'}, 'Location', 'best')
title('Outcomes', 'FontSize', 14)
xlabel('time')
ylabel('level')
axis square tight
grid on

end
