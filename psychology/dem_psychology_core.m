function DEM = dem_psychology_core(N, beliefPrior, M2V, M1V)
% Core active inference psychology simulation.
%
% The agent holds direct beliefs about energy, reward, and fatigue.
% No categorical self-model layer — beliefs are about the world.
%
% Args:
%   N            - number of time steps
%   beliefPrior  - 3x1 prior expectations [energy; reward; fatigue]
%   M2V          - prior precision exponent: M(2).V = exp(M2V)
%   M1V          - sensory precision exponent: M(1).V = exp(M1V)

% DEM parameters
%--------------------------------------------------------------------------
M(1).E.d = 1;                             % approximation order
M(1).E.n = 2;                             % embedding order
M(1).E.s = 1;                             % smoothness

% initialise hidden causes and action
%--------------------------------------------------------------------------
g            = psychology_expect([], beliefPrior, []);
action.energy = g.energy;

% generative process
%==========================================================================
G(1).g  = @(x, v, a, P) psychology_observe(x, v, a, P);
G(1).v  = psychology_observe([], [], action, action);
G(1).V  = exp(16);
G(1).U  = exp(2);
G(1).R  = ones(3, 1);
G(1).pE = action;

G(2).a  = spm_vec(action);
G(2).v  = 0;
G(2).V  = exp(16);

% generative model
%==========================================================================
M(1).g  = @(x, v, P) psychology_expect([], v, P);
M(1).v  = g;
M(1).V  = exp(M1V);
M(1).pE = 0;                              % no learnable parameters
M(1).pC = 0;

M(2).v  = beliefPrior;
M(2).V  = exp(M2V);

% causes and priors
%--------------------------------------------------------------------------
C = zeros(1, N);
U = repmat(beliefPrior, 1, N);

% assemble and solve
%--------------------------------------------------------------------------
DEM.M  = M;
DEM.G  = G;
DEM.C  = C;
DEM.U  = U;
DEM.db = 0;

DEM = spm_ADEM(DEM);

end
