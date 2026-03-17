function g = psychology_expect(x, v, P)
% Generative model: beliefs directly predict observations.
% v = [expected_energy; expected_reward; expected_fatigue]
% No categorical layer — beliefs are about the world, not about self-type.

g.energy  = v(1);
g.reward  = v(2);
g.fatigue = v(3);

end
