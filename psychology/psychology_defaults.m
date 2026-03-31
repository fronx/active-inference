function [P, x0, action, observation] = psychology_defaults()
% Shared parameters and templates for the psychology state-space model.

P = struct();

% smooth transforms and operating range
P.reserveMax     = 1.6;
P.effortMax      = 1.4;
P.softplusScale  = 4.0;

% state dynamics
P.reserveSetpoint = 1.0;
P.reserveRecover  = 0.32;
P.reserveSpend    = 0.58;
P.fatigueBuild    = 0.42;
P.fatigueDecay    = 0.18;
P.effortTau       = 1.10;
P.tonicEffortBaseline = -0.85;
P.actionEffortGain    = 1.60;

% mapping from beliefs and states to desired effort
P.driveOffset     = -1.05;
P.driveEnergy     = 1.10;
P.driveReward     = 1.55;
P.driveFatigue    = 1.65;
P.driveReserve    = 0.95;
P.driveStateFatigue = 1.90;

% observation model
P.rewardFatiguePenalty = 0.55;
P.observedFatigueLoad  = 0.45;
P.beliefWeightEnergy   = 0.55;
P.beliefWeightReward   = 0.60;
P.beliefWeightFatigue  = 0.60;

% initial conditions
x0 = [1.20; -1.60; -0.85];

% action template
action.energy = 0;

% observation template
observation.energy  = 0;
observation.reward  = 0;
observation.fatigue = 0;

end
