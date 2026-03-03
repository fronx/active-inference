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
rng('default')

N = 48;                                    % time steps


% generative process and model
%==========================================================================
M(1).E.d = 1;                             % approximation order
M(1).E.n = 2;                             % embedding order
M(1).E.s = 1;                             % smoothness


% priors (templates for each mood state)
%--------------------------------------------------------------------------
% Columns represent [manic, balanced, depressed].
% These define what each mood state "looks like" in terms of observables.
% Analogous to P.position and P.signal in morphogenesis.

P.energy  = [1.4; 1.0; 0.6];              % energy expenditure
P.reward  = [0.8; 0.5; 0.2];              % subjective reward
P.fatigue = [0.2; 0.4; 0.8];              % subjective fatigue


% initialise hidden causes, action, and expectations
%--------------------------------------------------------------------------
moodLogits = zeros(3, 1);                  % hidden causes (mood logits)
g          = expect([], moodLogits, P);    % initial expected observations
action.energy = g.energy;                  % initial action


% generative process
%==========================================================================

% level 1 of generative process
%--------------------------------------------------------------------------
G(1).g  = @(x, v, a, P) observe(x, v, a, P);
G(1).v  = observe([], [], action, action); % initial observations
G(1).V  = exp(16);                         % precision (low process noise)
G(1).U  = exp(2);                          % precision (action)
G(1).R  = ones(3, 1);                      % restriction: all obs inform action
G(1).pE = action;                          % action template

% level 2: action
%--------------------------------------------------------------------------
G(2).a  = spm_vec(action);                 % endogenous cause (action)
G(2).v  = 0;                               % exogenous cause
G(2).V  = exp(16);


% generative model
%==========================================================================

% level 1 of the generative model
%--------------------------------------------------------------------------
M(1).g  = @(x, v, P) expect([], v, P);
M(1).v  = g;                               % initial expected observations
M(1).V  = exp(3);                          % sensory precision (trusts evidence)
M(1).pE = P;                               % mood templates (prior expectation)
np      = spm_length(P);
M(1).pC = eye(np, np) * exp(-2);           % template flexibility (prior covariance)

% level 2: hidden causes
%--------------------------------------------------------------------------
M(2).v  = moodLogits;


% regime-specific settings
%==========================================================================
% M(2).v  = initial mood logits (where inference starts)
% U       = prior expectation on mood logits over time (the attractor)
% M(2).V  = precision on that prior (how rigid the attractor is)
% M(1).V  = sensory precision (how much evidence is trusted)

moodPrior = zeros(3, 1);                   % default: flat prior

switch regime
    case 'healthy'
        M(2).V = exp(-2);                  % loose prior: beliefs update freely
        % M(1).pC already set — templates revisable

    case 'depressed'
        moodPrior = [-1; 0; 2];            % prior biased toward depressive state
        M(2).v = moodPrior;                % start at the prior
        M(2).V = exp(4);                   % rigid prior: resists evidence
        M(1).V = exp(0);                   % low sensory precision
        M(1).pC = eye(np, np) * exp(-8);   % templates frozen

    case 'manic'
        moodPrior = [2; 0; -1];            % prior biased toward manic state
        M(2).v = moodPrior;                % start at the prior
        M(2).V = exp(4);                   % rigid prior
        M(1).V = exp(0);                   % low sensory precision
        M(1).pC = eye(np, np) * exp(-8);   % templates frozen
end


% hidden cause and prior expectations
%--------------------------------------------------------------------------
C = zeros(1, N);                           % exogenous causes (none)
U = repmat(moodPrior, 1, N);              % prior on mood logits (the attractor)


% assemble model structure
%--------------------------------------------------------------------------
DEM.M = M;
DEM.G = G;
DEM.C = C;
DEM.U = U;
DEM.db = 0;                                % skip in-loop diagnostic plots


% solve
%==========================================================================
DEM = spm_ADEM(DEM);


% Graphics
%==========================================================================
spm_figure('GetWin', ['Psychology: ' regime]); clf

% free energy
%--------------------------------------------------------------------------
subplot(2, 2, 1)
plot(-DEM.J)
title(sprintf('Free energy (%s)', regime), 'FontSize', 14)
xlabel('time')
ylabel('free energy')
axis square tight
grid on

% core beliefs over time
%--------------------------------------------------------------------------
subplot(2, 2, 2)
beliefs = zeros(3, N);
for t = 1:N
    v = DEM.qU.v{2}(:, t);
    beliefs(:, t) = spm_softmax(v);
end
plot(1:N, beliefs')
legend({'effort pays off', 'moderate', 'effort is futile'}, 'Location', 'best')
title('Core beliefs', 'FontSize', 14)
xlabel('time')
ylabel('P(belief)')
axis square tight
grid on

% energy expenditure (action) over time
%--------------------------------------------------------------------------
subplot(2, 2, 3)
energies = zeros(1, N);
for t = 1:N
    a = spm_unvec(DEM.qU.a{2}(:, t), action);
    energies(t) = a.energy;
end
plot(1:N, energies, 'LineWidth', 2)
hold on
plot([1 N], [1.0 1.0], '--k')
title('Energy expenditure', 'FontSize', 14)
xlabel('time')
ylabel('energy')
axis square tight
grid on

% actual reward vs fatigue
%--------------------------------------------------------------------------
subplot(2, 2, 4)
rewards  = zeros(1, N);
fatigues = zeros(1, N);
for t = 1:N
    a = spm_unvec(DEM.qU.a{2}(:, t), action);
    e = a.energy;
    rewards(t)  = e / (1 + abs(e));
    fatigues(t) = 0.3 * e^2;
end
plot(1:N, rewards, 'g', 'LineWidth', 2); hold on
plot(1:N, fatigues, 'r', 'LineWidth', 2)
legend({'reward', 'fatigue'}, 'Location', 'best')
title('Outcomes', 'FontSize', 14)
xlabel('time')
ylabel('level')
axis square tight
grid on

end  % main function


% Equations of motion and observer functions
%==========================================================================

% generative process: maps energy expenditure to actual observations
%--------------------------------------------------------------------------
function g = observe(x, v, action, P)
% Simple, symmetric physics. No psychology baked in.
% Reality has diminishing returns and quadratic fatigue cost.

action = spm_unvec(action, P);
energy = action.energy;

g.energy  = energy;
g.reward  = energy / (1 + abs(energy));    % diminishing returns
g.fatigue = 0.3 * energy.^2;              % quadratic cost

end


% generative model: maps mood beliefs to expected observations
%--------------------------------------------------------------------------
function g = expect(x, moodLogits, P)
% softmax(moodLogits) gives belief weights over [manic, balanced, depressed].
% Expected observation = template * beliefs (weighted sum).
% Directly parallels morphogenesis: P.position * identityBelief.

mood = spm_softmax(moodLogits);

g.energy  = P.energy'  * mood;
g.reward  = P.reward'  * mood;
g.fatigue = P.fatigue' * mood;

end
