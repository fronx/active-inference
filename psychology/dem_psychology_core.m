function DEM = dem_psychology_core(N, beliefPrior, M2V, M1V)
% Core active inference psychology simulation.
%
% The agent holds direct beliefs about energy, reward, and fatigue, but
% now acts through hidden states with temporal dynamics.
%
% Args:
%   N            - number of time steps
%   beliefPrior  - 3x1 prior expectations [energy; reward; fatigue]
%   M2V          - prior precision exponent: M(2).V = exp(M2V)
%   M1V          - sensory precision exponent: M(1).V = exp(M1V)

[P, x0, action, y0] = psychology_defaults();

% DEM parameters
%--------------------------------------------------------------------------
M(1).E.d = 2;                             % approximation order
M(1).E.n = 3;                             % embedding order
M(1).E.s = 1;                             % smoothness

% initialise process observations and action
%--------------------------------------------------------------------------
y0 = psychology_observe(x0, [], action, P);

% generative process
%==========================================================================
G(1).f  = @(x, v, a, P) psychology_state_process_f(x, v, a, P);
G(1).g  = @(x, v, a, P) psychology_observe(x, v, a, P);
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

% generative model
%==========================================================================
M(1).f  = @(x, v, P) psychology_state_model_f(x, v, P);
M(1).g  = @(x, v, P) psychology_expect(x, v, P);
M(1).x  = x0;
M(1).v  = y0;
M(1).V  = exp(M1V);
M(1).W  = exp(1);
M(1).pE = P;                              % fixed parameters shared with process
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
DEM.P = P;

end
