function export_experiment_004_figure(outfile)
% Export the felt-energy observation split figure for the docs.

if nargin < 1 || isempty(outfile)
    outfile = fullfile('docs', 'experiments', 'assets', '004-felt-energy-observation.png');
end

addpath('spm12');
addpath('spm12/toolbox/DEM');
addpath('psychology');

outdir = fileparts(outfile);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

set(0, 'defaultfigurevisible', 'off');

regimes = {'healthy', 'depressed', 'manic'};
fig = figure('visible', 'off', 'position', [100 100 1500 1100]);

for i = 1:numel(regimes)
    [N, beliefPrior, M2V, M1V] = psychology_params(regimes{i});
    DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);
    traces = psychology_extract(DEM, N);

    subplot(3, 3, 3 * (i - 1) + 1)
    plot(1:N, traces.feltEnergy, 'Color', [0.49 0.23 0.93], 'LineWidth', 2); hold on
    plot(1:N, traces.energy, 'k', 'LineWidth', 1.75)
    plot(1:N, traces.opportunity, '-.b', 'LineWidth', 1.0)
    hold off
    title(sprintf('%s felt vs output', regimes{i}))
    axis tight
    grid on

    subplot(3, 3, 3 * (i - 1) + 2)
    plot(1:N, traces.reserves, '--b', 'LineWidth', 1.5); hold on
    plot(1:N, traces.capacity, 'c', 'LineWidth', 2)
    plot(1:N, traces.fatigueState, '--r', 'LineWidth', 1.5)
    hold off
    title(sprintf('%s body state', regimes{i}))
    axis tight
    grid on

    subplot(3, 3, 3 * (i - 1) + 3)
    plot(1:N, traces.reward, 'g', 'LineWidth', 2); hold on
    plot(1:N, traces.fatigue, 'r', 'LineWidth', 2)
    plot(1:N, traces.actionTarget, ':k', 'LineWidth', 1.5)
    hold off
    title(sprintf('%s value and cost', regimes{i}))
    axis tight
    grid on
end

print(fig, outfile, '-dpng', '-r160');
close(fig);
end
