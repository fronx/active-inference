function [P, x0, action, observation] = psychology_defaults()
% Shared parameters and templates for the psychology pacing model.

P = struct();

% smooth transforms and operating range
P.reserveMax       = 1.45;
P.activationMax    = 1.20;
P.softplusScale    = 3.50;

% opportunity pulses
P.opportunityBase   = 0.04;
P.opportunityAmp    = 0.96;
P.opportunityWidth  = 0.045;
P.opportunityCenters = [0.10 0.30 0.50 0.70 0.90];
P.opportunityBeliefBase = 0.18;

% state dynamics
P.reserveSetpoint        = 1.00;
P.reserveRecover         = 0.46;
P.reserveSpend           = 0.82;
P.fatigueBuild           = 0.72;
P.fatigueDecay           = 0.24;
P.activationTau          = 0.55;
P.activationBaseline     = -1.15;
P.activationOpportunity  = 2.20;
P.activationFatigueDrag  = 1.90;
P.activationReserveDrag  = 1.20;
P.actionActivationGain   = 2.40;

% belief influence on the agent's own expected activation dynamics
P.modelDriveEnergy   = 0.70;
P.modelDriveReward   = 1.30;
P.modelDriveFatigue  = 1.95;

% observation model
P.rewardFatiguePenalty = 0.30;
P.observedFatigueLoad  = 0.35;
P.beliefWeightEnergy   = 0.35;
P.beliefWeightReward   = 0.55;
P.beliefWeightFatigue  = 0.10;

% initial conditions: high reserves, low fatigue, low activation
x0 = [1.15; -2.10; -1.45];

% action template
action.energy = 0;

% observation template
observation.energy      = 0;
observation.reward      = 0;
observation.fatigue     = 0;
observation.opportunity = 0;

end
