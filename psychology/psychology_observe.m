function g = psychology_observe(x, v, action, P)
% Generative process: latent reserves and fatigue shape what the action
% actually produces over time.

[~, actionTemplate, ~, ~] = deal([]);
[~, ~, actionTemplate, ~] = psychology_defaults();
if isempty(action)
    action = actionTemplate;
end
action = spm_unvec(action, actionTemplate);

[~, g] = psychology_state_unpack(x, P, P.actionEffortGain * action.energy);

end
