function g = psychology_expect(x, v, P)
% Generative model: hidden states predict observations, while beliefs shape
% the state dynamics through M(1).f.

[~, stateObs] = psychology_state_unpack(x, P, 0);

availability = P.opportunityBeliefBase ...
    + (1 - P.opportunityBeliefBase) * stateObs.opportunity;

g.energy      = (1 - P.beliefWeightEnergy) * stateObs.energy ...
    + P.beliefWeightEnergy * availability * v(1);
g.reward      = (1 - P.beliefWeightReward) * stateObs.reward ...
    + P.beliefWeightReward * stateObs.opportunity * v(2);
g.fatigue     = (1 - P.beliefWeightFatigue) * stateObs.fatigue ...
    + P.beliefWeightFatigue * v(3);
g.opportunity = stateObs.opportunity;

end
