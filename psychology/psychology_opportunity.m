function o = psychology_opportunity(P, tau)
% Deterministic stream of opportunities over normalized time tau in [0, 1].

global t

if nargin < 2 || isempty(tau)
    try
        tau = t;
    catch
        tau = 0;
    end
end

if isempty(tau)
    tau = 0;
end

pulseMass = 0;
for i = 1:numel(P.opportunityCenters)
    d = (tau - P.opportunityCenters(i)) / P.opportunityWidth;
    pulseMass = pulseMass + exp(-0.5 * d.^2);
end

o = P.opportunityBase + P.opportunityAmp * (1 - exp(-pulseMass));

end
