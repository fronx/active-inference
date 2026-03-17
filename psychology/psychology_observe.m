function g = psychology_observe(x, v, action, P)
% Generative process: maps energy expenditure to actual observations.
% Simple, symmetric physics. No psychology baked in.

action = spm_unvec(action, P);
energy = action.energy;

g.energy  = energy;
g.reward  = energy / (1 + abs(energy));    % diminishing returns
g.fatigue = 0.3 * energy.^2;              % quadratic cost

end
