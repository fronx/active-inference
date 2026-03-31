function g = psychology_expect(x, v, P)
% Generative model: hidden states predict observations, while beliefs shape
% the state dynamics through M(1).f.

[~, stateObs] = psychology_state_unpack(x, P);

g.energy  = (1 - P.beliefWeightEnergy) * stateObs.energy ...
    + P.beliefWeightEnergy * v(1);
g.reward  = (1 - P.beliefWeightReward) * stateObs.reward ...
    + P.beliefWeightReward * v(2);
g.fatigue = (1 - P.beliefWeightFatigue) * stateObs.fatigue ...
    + P.beliefWeightFatigue * v(3);

end
