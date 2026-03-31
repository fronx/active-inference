function export_experiment_figures()
% Export experiment figures for the psychology docs.

addpath('spm12');
addpath('spm12/toolbox/DEM');
addpath('psychology');

assets_dir = fullfile('docs', 'experiments', 'assets');
if ~exist(assets_dir, 'dir')
    mkdir(assets_dir);
end

set(0, 'defaultfigurevisible', 'off');

export_experiment_001(fullfile(assets_dir, '001-state-dynamics.png'));
export_experiment_003(fullfile(assets_dir, '003-body-self-model-split.png'));
export_experiment_004_figure(fullfile(assets_dir, '004-felt-energy-observation.png'));
export_experiment_005_figure(fullfile(assets_dir, '005-body-limited-mobilization.png'));
export_experiment_006_figure(fullfile(assets_dir, '006-felt-energy-damping.png'));

end

function export_experiment_001(outfile)
regimes = {'healthy', 'depressed', 'manic'};

fig = figure('visible', 'off', 'position', [100 100 1500 1100]);

for i = 1:numel(regimes)
    [N, beliefPrior, M2V, M1V] = psychology_params(regimes{i});
    DEM = legacy_dem_core(N, beliefPrior, M2V, M1V);
    traces = legacy_extract(DEM, N);

    subplot(3, 3, 3 * (i - 1) + 1)
    plot(1:N, traces.beliefs', 'LineWidth', 1.25)
    title(sprintf('%s beliefs', regimes{i}))
    axis tight
    grid on

    subplot(3, 3, 3 * (i - 1) + 2)
    plot(1:N, traces.energy, 'k', 'LineWidth', 2); hold on
    plot(1:N, traces.reserves, '--b', 'LineWidth', 1.5)
    plot(1:N, traces.effort, ':m', 'LineWidth', 1.5)
    hold off
    title(sprintf('%s state dynamics', regimes{i}))
    axis tight
    grid on

    subplot(3, 3, 3 * (i - 1) + 3)
    plot(1:N, traces.reward, 'g', 'LineWidth', 2); hold on
    plot(1:N, traces.fatigue, 'r', 'LineWidth', 2)
    plot(1:N, traces.fatigueState, '--k', 'LineWidth', 1.5)
    hold off
    title(sprintf('%s outcomes', regimes{i}))
    axis tight
    grid on
end

print(fig, outfile, '-dpng', '-r160');
close(fig);
end

function export_experiment_002(outfile)
regimes = {'healthy', 'depressed', 'manic'};

fig = figure('visible', 'off', 'position', [100 100 1500 1100]);

for i = 1:numel(regimes)
    [N, beliefPrior, M2V, M1V] = psychology_params(regimes{i});
    DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);
    traces = psychology_extract(DEM, N);

    subplot(3, 3, 3 * (i - 1) + 1)
    plot(1:N, traces.beliefs', 'LineWidth', 1.25)
    title(sprintf('%s beliefs', regimes{i}))
    axis tight
    grid on

    subplot(3, 3, 3 * (i - 1) + 2)
    plot(1:N, traces.energy, 'k', 'LineWidth', 2); hold on
    plot(1:N, traces.reserves, '--b', 'LineWidth', 1.5)
    plot(1:N, traces.activation, ':m', 'LineWidth', 1.5)
    plot(1:N, traces.opportunity, '-.k', 'LineWidth', 1.0)
    hold off
    title(sprintf('%s pacing', regimes{i}))
    axis tight
    grid on

    subplot(3, 3, 3 * (i - 1) + 3)
    plot(1:N, traces.reward, 'g', 'LineWidth', 2); hold on
    plot(1:N, traces.fatigue, 'r', 'LineWidth', 2)
    plot(1:N, traces.fatigueState, '--k', 'LineWidth', 1.5)
    hold off
    title(sprintf('%s outcomes', regimes{i}))
    axis tight
    grid on
end

print(fig, outfile, '-dpng', '-r160');
close(fig);
end

function export_experiment_003(outfile)
regimes = {'healthy', 'depressed', 'manic'};

fig = figure('visible', 'off', 'position', [100 100 1500 1100]);

for i = 1:numel(regimes)
    [N, beliefPrior, M2V, M1V] = psychology_params(regimes{i});
    DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);
    traces = psychology_extract(DEM, N);

    subplot(3, 3, 3 * (i - 1) + 1)
    plot(1:N, traces.energy, 'k', 'LineWidth', 2); hold on
    plot(1:N, traces.activation, ':m', 'LineWidth', 1.5)
    plot(1:N, traces.opportunity, '-.b', 'LineWidth', 1.0)
    hold off
    title(sprintf('%s output', regimes{i}))
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

function DEM = legacy_dem_core(N, beliefPrior, M2V, M1V)
[P, x0, action] = legacy_defaults();
y0 = legacy_observe(x0, [], action, P);

M(1).E.d = 2;
M(1).E.n = 3;
M(1).E.s = 1;

G(1).f  = @(x, v, a, P) legacy_process_f(x, v, a, P);
G(1).g  = @(x, v, a, P) legacy_observe(x, v, a, P);
G(1).x  = x0;
G(1).v  = y0;
G(1).V  = exp(16);
G(1).W  = exp(16);
G(1).U  = exp(2);
G(1).R  = ones(3, 1);
G(1).pE = P;

G(2).a  = spm_vec(action);
G(2).v  = 0;
G(2).V  = exp(16);

M(1).f  = @(x, v, P) legacy_model_f(x, v, P);
M(1).g  = @(x, v, P) legacy_expect(x, v, P);
M(1).x  = x0;
M(1).v  = y0;
M(1).V  = exp(M1V);
M(1).W  = exp(1);
M(1).pE = P;
M(1).pC = 0;

M(2).v  = beliefPrior;
M(2).V  = exp(M2V);

DEM.M  = M;
DEM.G  = G;
DEM.C  = zeros(1, N);
DEM.U  = repmat(beliefPrior, 1, N);
DEM.db = 0;

DEM = spm_ADEM(DEM);
DEM.P = P;
end

function traces = legacy_extract(DEM, N)
[P, ~, action] = legacy_defaults();

traces.beliefs      = zeros(3, N);
traces.energy       = zeros(1, N);
traces.reward       = zeros(1, N);
traces.fatigue      = zeros(1, N);
traces.reserves     = zeros(1, N);
traces.effort       = zeros(1, N);
traces.fatigueState = zeros(1, N);

for t = 1:N
    traces.beliefs(:, t) = DEM.qU.v{2}(:, t);
    a = spm_unvec(DEM.qU.a{2}(:, t), action);
    y = spm_unvec(DEM.pU.v{1}(:, t), struct('energy', 0, 'reward', 0, 'fatigue', 0));
    [state, ~] = legacy_unpack(DEM.pU.x{1}(:, t), P, P.actionEffortGain * a.energy);

    traces.energy(t)       = y.energy;
    traces.reward(t)       = y.reward;
    traces.fatigue(t)      = y.fatigue;
    traces.reserves(t)     = state.reserves;
    traces.effort(t)       = state.effort;
    traces.fatigueState(t) = state.fatigueState;
end
end

function [P, x0, action] = legacy_defaults()
P = struct();
P.reserveMax          = 1.6;
P.effortMax           = 1.4;
P.softplusScale       = 4.0;
P.reserveSetpoint     = 1.0;
P.reserveRecover      = 0.32;
P.reserveSpend        = 0.58;
P.fatigueBuild        = 0.42;
P.fatigueDecay        = 0.18;
P.effortTau           = 1.10;
P.tonicEffortBaseline = -0.85;
P.actionEffortGain    = 1.60;
P.driveOffset         = -1.05;
P.driveEnergy         = 1.10;
P.driveReward         = 1.55;
P.driveFatigue        = 1.65;
P.driveReserve        = 0.95;
P.driveStateFatigue   = 1.90;
P.rewardFatiguePenalty = 0.55;
P.observedFatigueLoad  = 0.45;
P.beliefWeightEnergy   = 0.55;
P.beliefWeightReward   = 0.60;
P.beliefWeightFatigue  = 0.60;

x0 = [1.20; -1.60; -0.85];
action.energy = 0;
end

function dx = legacy_model_f(x, v, P)
[state, ~] = legacy_unpack(x, P, 0);
target = P.driveOffset + P.driveEnergy * v(1) + P.driveReward * v(2) ...
    - P.driveFatigue * v(3) + P.driveReserve * state.reserves ...
    - P.driveStateFatigue * state.fatigueState;

dx    = zeros(3, 1);
dx(1) = P.reserveRecover * (P.reserveSetpoint - state.reserves) - P.reserveSpend * state.energy;
dx(2) = P.fatigueBuild * state.energy.^2 - P.fatigueDecay * state.fatigueState;
dx(3) = (target - x(3)) / P.effortTau;
end

function dx = legacy_process_f(x, v, a, P)
[~, ~, action] = legacy_defaults();
a = spm_unvec(a, action);
[state, ~] = legacy_unpack(x, P, P.actionEffortGain * a.energy);

dx    = zeros(3, 1);
dx(1) = P.reserveRecover * (P.reserveSetpoint - state.reserves) - P.reserveSpend * state.energy;
dx(2) = P.fatigueBuild * state.energy.^2 - P.fatigueDecay * state.fatigueState;
dx(3) = (P.tonicEffortBaseline - x(3)) / P.effortTau;
end

function g = legacy_expect(x, v, P)
[~, obs] = legacy_unpack(x, P, 0);
g.energy  = (1 - P.beliefWeightEnergy) * obs.energy + P.beliefWeightEnergy * v(1);
g.reward  = (1 - P.beliefWeightReward) * obs.reward + P.beliefWeightReward * v(2);
g.fatigue = (1 - P.beliefWeightFatigue) * obs.fatigue + P.beliefWeightFatigue * v(3);
end

function g = legacy_observe(x, v, action, P)
[~, ~, actionTemplate] = legacy_defaults();
action = spm_unvec(action, actionTemplate);
[~, g] = legacy_unpack(x, P, P.actionEffortGain * action.energy);
end

function [state, obs] = legacy_unpack(x, P, effortShift)
state.reserves     = P.reserveMax * (1 ./ (1 + exp(-x(1))));
state.fatigueState = log1p(exp(P.softplusScale * x(2))) / P.softplusScale;
state.effort       = P.effortMax * (1 ./ (1 + exp(-(x(3) + effortShift))));
state.energy       = state.reserves * state.effort;

obs.energy  = state.energy;
obs.reward  = state.energy / (1 + state.energy) - P.rewardFatiguePenalty * state.fatigueState;
obs.fatigue = state.fatigueState + P.observedFatigueLoad * state.energy.^2;
end
