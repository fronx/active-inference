function export_experiment_005_figure(outfile)
% Export the body-limited mobilization figure for the docs.

if nargin < 1 || isempty(outfile)
    outfile = fullfile('docs', 'experiments', 'assets', '005-body-limited-mobilization.png');
end

export_experiment_004_figure(outfile);
end
