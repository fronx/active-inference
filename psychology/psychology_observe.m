function g = psychology_observe(x, v, action, P)
% Generative process: latent reserves and fatigue shape what the action
% actually produces over time.

[~, ~, ~, ~, actionTemplate, ~] = psychology_defaults();
if isempty(action)
    action = actionTemplate;
end
action = spm_unvec(action, actionTemplate);

[state0, ~] = psychology_state_unpack(x, P, 0);
bodyLeverage = min(state0.reserves / max(state0.capacity, exp(-8)), 1);
[~, g] = psychology_state_unpack(x, P, ...
    P.actionActivationGain * action.energy * bodyLeverage);

end
