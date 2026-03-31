function export_experiment_006_figure(outfile)
% Export the felt-energy damping figure for the docs.

if nargin < 1 || isempty(outfile)
    outfile = fullfile('docs', 'experiments', 'assets', '006-felt-energy-damping.png');
end

export_experiment_004_figure(outfile);
end
