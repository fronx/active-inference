function psychology_worker(fifo_path)
% Persistent Octave worker. Reads JSON params from a FIFO, writes results to stdout.
% Stays alive between requests to avoid startup cost.
%
% Args:
%   fifo_path - path to a named pipe (FIFO) for receiving commands
%
% Protocol:
%   - Reads one line of JSON from FIFO
%   - Runs simulation
%   - Writes one line of JSON to stdout
%   - Writes __DONE__ on its own line
%   - Loops until FIFO is closed

% signal ready
fprintf('__READY__\n');
fflush(stdout);

while true
    % open FIFO for reading (blocks until writer connects)
    fid = fopen(fifo_path, 'r');
    if fid == -1; break; end

    line = fgetl(fid);
    fclose(fid);

    if ~ischar(line); continue; end
    line = strtrim(line);
    if isempty(line); continue; end

    try
        params = jsondecode(line);

        beliefPrior = params.beliefPrior(:);
        N           = params.N;
        M2V         = params.M2V;
        M1V         = params.M1V;

        DEM = dem_psychology_core(N, beliefPrior, M2V, M1V);
        traces = psychology_extract(DEM, N);

        result.timesteps        = 1:N;
        result.freeEnergy       = -DEM.J;
        result.beliefs.energy   = traces.beliefs(1, :);
        result.beliefs.reward   = traces.beliefs(2, :);
        result.beliefs.fatigue  = traces.beliefs(3, :);
        result.energy           = traces.energy;
        result.reward           = traces.reward;
        result.fatigue          = traces.fatigue;
        result.reserves         = traces.reserves;
        result.activation       = traces.activation;
        result.effort           = traces.effort;
        result.fatigueState     = traces.fatigueState;
        result.actionTarget     = traces.actionTarget;
        result.opportunity      = traces.opportunity;

        fprintf('%s\n', jsonencode(result));
    catch err
        fprintf('%s\n', jsonencode(struct('error', err.message)));
    end

    fprintf('__DONE__\n');
    fflush(stdout);
end

end
