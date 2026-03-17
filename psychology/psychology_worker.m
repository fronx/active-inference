function psychology_worker()
% Persistent Octave worker. Reads JSON params from stdin, writes JSON results to stdout.
% Stays alive between requests to avoid startup cost.
%
% Protocol:
%   - Reads one line of JSON from stdin
%   - Runs simulation
%   - Writes one line of JSON to stdout
%   - Writes __DONE__ on its own line
%   - Loops until stdin closes

% signal ready
fprintf('__READY__\n');
fflush(stdout);

while true
    line = fgetl(stdin);
    if ~ischar(line); break; end        % EOF — exit
    line = strtrim(line);
    if isempty(line); continue; end     % skip blank lines

    try
        params = jsondecode(line);

        beliefPrior = params.beliefPrior(:);
        N           = params.N;
        M2V         = params.M2V;
        M1V         = params.M1V;

        DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);

        action.energy = 0;
        [beliefs, energies, rewards, fatigues] = psychology_extract(DEM, N, action);

        result.timesteps        = 1:N;
        result.freeEnergy       = -DEM.J;
        result.beliefs.energy   = beliefs(1, :);
        result.beliefs.reward   = beliefs(2, :);
        result.beliefs.fatigue  = beliefs(3, :);
        result.energy  = energies;
        result.reward  = rewards;
        result.fatigue = fatigues;

        fprintf('%s\n', jsonencode(result));
    catch err
        fprintf('%s\n', jsonencode(struct('error', err.message)));
    end

    fprintf('__DONE__\n');
    fflush(stdout);
end

end
