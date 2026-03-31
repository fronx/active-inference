function [PG, PM, xG0, xM0, action, observation] = psychology_defaults(beliefPrior)
% Shared setup for the psychology model.
%
% PG  - generative process parameters (body reality)
% PM  - generative model parameters (self-model of the body)
% xG0 - initial hidden states of the process
% xM0 - initial hidden states expected by the model

if nargin < 1 || isempty(beliefPrior)
    beliefPrior = [1.0; 0.5; 0.3];
end

bias = psychology_regime_bias(beliefPrior);
maniaBias = max(bias, 0);
depressionBias = max(-bias, 0);

% generative process: physical body and environment
PG = struct();

PG.softplusScale      = 2.6;
PG.activationMax      = 1.55;

PG.opportunityBase    = 0.04;
PG.opportunityAmp     = 0.96;
PG.opportunityWidth   = 0.045;
PG.opportunityCenters = [0.10 0.30 0.50 0.70 0.90];
PG.opportunityBeliefBase = 0.18;

PG.reserveRecover        = 0.48;
PG.reserveSpend          = 0.95;
PG.fatigueBuild          = 0.88;
PG.fatigueDecay          = 0.14;
PG.activationTau         = 0.40;
PG.activationBaseline    = -1.45;
PG.activationOpportunity = 2.10;
PG.activationFatigueDrag = 1.80;
PG.activationReserveDrag = 1.35;
PG.actionActivationGain  = 2.05;

PG.capacityBaseline      = 1.05;
PG.capacityHomeostasis   = 0.05;
PG.capacityBuild         = 0.02;
PG.capacityAtrophy       = 0.12;
PG.capacityDamage        = 0.18;
PG.moderateUseCenter     = 0.30;
PG.moderateUseWidth      = 0.10;
PG.underuseThreshold     = 0.10;
PG.underuseScale         = 0.04;
PG.overuseThreshold      = 0.40;
PG.overuseScale          = 0.06;

PG.rewardFatiguePenalty  = 0.09;
PG.observedFatigueLoad   = 0.24;
PG.feltEnergyOpportunityBoost = 0.14;
PG.feltEnergyFatiguePenalty   = 0.16;
PG.feltEnergyBodyFloor        = 0.25;

% process initial conditions: different bodies for different phenotypes
capacity0   = 1.00 + 0.35 * maniaBias - 0.35 * depressionBias;
reserves0   = capacity0 * (0.82 + 0.18 * maniaBias - 0.34 * depressionBias);
fatigue0    = 0.12 + 0.08 * maniaBias + 0.18 * depressionBias;
activation0 = 0.10 + 0.14 * maniaBias - 0.10 * depressionBias;

% generative model: beliefs about the body
PM = PG;
PM.activationOpportunity = PG.activationOpportunity * exp(0.70 * bias);
PM.activationFatigueDrag = PG.activationFatigueDrag * exp(-1.00 * bias);
PM.activationReserveDrag = PG.activationReserveDrag * exp(-0.60 * bias);
PM.reserveRecover        = PG.reserveRecover        * exp(0.16 * bias);
PM.capacityBuild         = PG.capacityBuild         * exp(0.18 * bias);
PM.capacityAtrophy       = PG.capacityAtrophy       * exp(-0.24 * bias);
PM.capacityDamage        = PG.capacityDamage        * exp(-0.34 * bias);

PM.modelDriveEnergy      = 0.75;
PM.modelDriveReward      = 1.45;
PM.modelDriveFatigue     = 2.10 - 1.30 * bias;
PM.beliefWeightEnergy    = 0.38;
PM.beliefWeightReward    = 0.54;
PM.beliefWeightFatigue   = 0.18;

% model initial conditions: self-estimate of the body
capacityM   = capacity0 * exp(0.24 * bias);
reservesM   = reserves0 * exp(0.18 * bias);
fatigueM    = fatigue0  * exp(-0.28 * bias);
activationM = activation0 + 0.02 * bias;

xG0 = [inv_softplus(reserves0, PG.softplusScale);
       inv_softplus(fatigue0, PG.softplusScale);
       inv_sigmoid(activation0 / PG.activationMax);
       inv_softplus(capacity0, PG.softplusScale)];

xM0 = [inv_softplus(reservesM, PM.softplusScale);
       inv_softplus(fatigueM, PM.softplusScale);
       inv_sigmoid(clamp01(activationM / PM.activationMax));
       inv_softplus(capacityM, PM.softplusScale)];

action.energy = 0;

observation.energy      = 0;
observation.feltEnergy  = 0;
observation.reward      = 0;
observation.fatigue     = 0;
observation.opportunity = 0;

end

function z = inv_sigmoid(y)
y = clamp01(y);
z = log(y ./ (1 - y));
end

function z = inv_softplus(y, scale)
y = max(y, exp(-8));
z = log(exp(scale * y) - 1) / scale;
end

function y = clamp01(x)
y = min(max(x, 1e-4), 1 - 1e-4);
end
