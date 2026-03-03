function DEM = DEM_morphogenesis
% This routine illustrates self-assembly or more for genesis under active
% inference (free energy minimisation).  It exploits the fact that one can
% express a systems (marginal) Lyapunov function in terms of a variational
% free energy.  This means that one can prescribe an attracting set in
% terms of the generative model that defines variational free energy.  In
% this example, the attracting set is a point attractor in the phase space
% of a multi-celled organism: where the states correspond to the location
% and (chemotactic) signal expression of 16 cells.  The generative model
% and process are remarkably simple; however, the ensuing migration and
% differentiation of the 16 cells illustrates self-assembly - in the sense
% that each cell starts of in the same location and releasing the same
% signals.  In essence, the systems dynamics rest upon each cell inferring
% its unique identity (in relation to all others) and behaving in accord
% with those inferences; in other words, inferring its place in the
% assembly and behaving accordingly.  Note that in this example there are
% no hidden states and everything is expressed in terms of hidden causes
% (because the attracting set is a point attractor)  Graphics are produced
% illustrating the morphogenesis using colour codes to indicate the cell
% type - that is interpreted in terms of genetic and epigenetic
% processing.
% _________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: DEM_morphogenesis.m 7679 2019-10-24 15:54:07Z spm $
 
 
% preliminaries
%--------------------------------------------------------------------------
clear global
rng('default')

% schema-tracing (deduplicated) -------------------------------------------
global TRACE_SCHEMA TRACE_SCHEMA_KEYS
TRACE_SCHEMA      = {};
TRACE_SCHEMA_KEYS = struct();

SPLIT    = 0;                              % split: 1 = upper, 2 = lower
N        = 32;                             % length of process (bins)
 
% generative process and model
%==========================================================================
M(1).E.d  = 1;                             % approximation order
M(1).E.n  = 2;                             % embedding order
M(1).E.s  = 1;                             % smoothness
 
% priors (prototype)
%--------------------------------------------------------------------------
L     = 2;
if L == 2
    Target =[0 0 2 0 0 0 0 0 0 0 0;
             0 0 0 0 1 0 0 0 0 0 0;
             2 0 0 0 0 0 4 0 4 0 3;
             0 0 0 0 1 0 0 0 0 0 0;
             0 0 2 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0];
end
 
if L == 4
    Target =[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0;
             0 0 0 2 0 0 0 0 0 3 0 0 0 4 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 4 0 1 0 0;
             0 0 0 2 0 0 0 0 0 3 0 0 0 4 0 0 0 0 0 0 0 1;
             0 0 0 0 0 0 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
end
 
                                      % fronx:
p(:,:,1) = Target > 0;                %   any cell (not used for differentiation)
p(:,:,2) = Target == 2 | Target == 1; %   red color channel
p(:,:,3) = Target == 3 | Target == 1; %   green color channel
p(:,:,4) = Target == 4;               %   blue color channel

[y,x] = find(p(:,:,1));     % cell positions

% fronx: spm_detrend subtracts the mean from the position coordinates
%        the apostrophe transposes the matrix to have 2 rows (x and y) and n columns (cells)
%        dividing by 2 scales the coordinates to fit better within the plotting area
P.position = spm_detrend([x(:) y(:)])'/2;
 
% signalling of each cell type
%--------------------------------------------------------------------------
n     = size(P.position,2);                      % number of cells
m     = size(p,3);                        % number of signals
j     = find(p(:,:,1));
for i = 1:m
    signal        = p(:,:,i);
    P.signal(i,:) = signal(j);
end

P.signal = double(P.signal);
P.sense   = morphogenesis(P.position, P.signal); % signal sensed at each position
 
% initialise action and expectations
%--------------------------------------------------------------------------
identityLogits = randn(n,n)/8;            % hidden causes (identity logits)
g = expect([],identityLogits,P);
action.position = g.position;                  % action (chemotaxis)
action.signal   = g.signal;                    % action (signal release)
 
 
% generative process 
%==========================================================================
R     = spm_cat({kron(eye(n,n),ones(2,2)) []; [] kron(eye(n,n),ones(4,4));
                 kron(eye(n,n),ones(4,2)) kron(eye(n,n),ones(4,4))});

% level 1 of generative process
%--------------------------------------------------------------------------
G(1).g  = @(position, v, action, P) observe(position, v, action, P);
G(1).v  = observe([], [], action, action);
G(1).V  = exp(16);                         % precision (noise)
G(1).U  = exp(2);                          % precision (action)
G(1).R  = R;                               % restriction matrix
G(1).pE = action;                          % form (action)
 
 
% level 2; causes (action)
%--------------------------------------------------------------------------
G(2).a  = spm_vec(action);                  % endogenous cause (action)
G(2).v  = 0;                               % exogenous  cause
G(2).V  = exp(16);
 
 
% generative model
%==========================================================================
 
% level 1 of the generative model: 
%--------------------------------------------------------------------------
M(1).g  = @(position, v, P) expect([], v, P);
M(1).v  = g;
M(1).V  = exp(3);
M(1).pE = P;
 
% level 2: 
%--------------------------------------------------------------------------
M(2).v  = identityLogits;
M(2).V  = exp(-2);
 
 
% hidden cause and prior identity expectations (and time)
%--------------------------------------------------------------------------
U     = zeros(n*n,N);
C     = zeros(1,N);
 
% assemble model structure
%--------------------------------------------------------------------------
DEM.M = M;
DEM.G = G;
DEM.C = C;
DEM.U = U;
 
% solve
%==========================================================================
DEM   = spm_ADEM(DEM);
spm_DEM_qU(DEM.qU,DEM.pU);


% split half simulations
%==========================================================================
if SPLIT
    
    % select (partially diferentiated cells to duplicate
    %----------------------------------------------------------------------
    t    = 8;
    v    = spm_unvec(DEM.pU.v{1}(:,t),DEM.M(1).v);
    if SPLIT > 1
        [i j] = sort(v.x(1,:), 'ascend');
    else
        [i j] = sort(v.x(1,:),'descend');
    end
    j    = [j(1:n/2) j(1:n/2)];
    
    % reset hidden causes and expectations
    %----------------------------------------------------------------------
    v    = spm_unvec(DEM.qU.v{2}(:,t),DEM.M(2).v);
    g    = spm_unvec(DEM.qU.v{1}(:,t),DEM.M(1).v);
    a    = spm_unvec(DEM.qU.a{2}(:,t),DEM.G(1).pE);
    
    v    = v(:,j);
    g.position  = g.position(:,j);
    g.signal    = g.signal(:,j);
    g.sense  = g.sense(:,j);
    action.position = action.position(:,j) + randn(size(action.position))/512;
    action.signal   = action.signal(:,j) + randn(size(action.signal))/512;
    
    DEM.M(1).v = g;
    DEM.M(2).v = v;
    DEM.G(2).a = spm_vec(a);
    
    % solve
    %----------------------------------------------------------------------
    DEM   = spm_ADEM(DEM);
    spm_DEM_qU(DEM.qU,DEM.pU);
    
end



 
% Graphics
%==========================================================================
 
% expected signal concentrations
%--------------------------------------------------------------------------
subplot(2,2,2); cla
A     = max(abs(P.position(:)))*3/2;
h     = 2/3;
 
x     = linspace(-A,A,32);
[x,y] = ndgrid(x,x);
position     = spm_detrend([x(:) y(:)])';
c     = morphogenesis(P.position, P.signal, position);
c     = c - min(c(:));
c     = c/max(c(:));
for i = 1:size(c,2)
    col = c(end - 2:end,i);
    plot(position(2,i),position(1,i),'.','markersize',32,'color',col); hold on
end
 
title('target signal','Fontsize',16)
xlabel('location')
ylabel('location')
set(gca,'Color','k');
axis([-1 1 -1 1]*A*(1+1/16))
axis square, box off
 
 
% free energy and expectations
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1'); clf
colormap pink
subplot(2,2,1); cla
 
plot(-DEM.J)
title('Free energy','Fontsize',16)
xlabel('time')
ylabel('Free energy')
axis square tight
grid on
 
subplot(2,2,2); cla
v      = spm_unvec(DEM.qU.v{2}(:,end),DEM.M(2).v);
[i j]  = max(v);
v(:,j) = v;
imagesc(spm_softmax(v))
title('softmax expectations','Fontsize',16)
xlabel('cell')
ylabel('cell')
axis square tight
 
 
% target morphology
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 2'); clf

subplot(2,2,1); cla
for i = 1:m
    for j = 1:n
        x = P.position(2,j);
        y = P.position(1,j) + i/6;
        if P.signal(i,j)
            plot(x,y,'.','markersize',24,'color','k'); hold on
        else
            plot(x,y,'.','markersize',24,'color','c'); hold on
        end
    end
end
xlabel('cell')
title('Encoding','Fontsize',16)
axis image off
hold off
 
subplot(2,2,2); cla
for i = 1:n
    position = P.position(:,i);
    sense = P.signal(end - 2:end,i);
    sense = full(max(min(sense,1),0));
    plot(position(2),position(1),'.','markersize',16,'color',sense);   hold on
    plot(position(2),position(1),'h','markersize',12,'color',h*sense); hold on
end
 
title('morphogenesis','Fontsize',16)
xlabel('location')
ylabel('location')
set(gca,'Color','k');
axis([-1 1 -1 1]*A)
axis square, box off
hold off
 
 
% graphics
%--------------------------------------------------------------------------
subplot(2,2,3); cla;
for t = 1:N
    v = spm_unvec(DEM.qU.a{2}(:,t),action);
    for i = 1:n
        pos = v.position(1,i);
        col = v.signal(end - 2:end,i);
        col = full(max(min(col,1),0));
        plot(t,pos,'.','markersize',16,'color',col); hold on
    end
end
 
title('morphogenesis','Fontsize',16)
xlabel('time')
ylabel('location')
set(gca,'Color','k');
set(gca,'YLim',[-1 1]*A)
axis square, box off
hold off
 
% movies
%--------------------------------------------------------------------------
subplot(2,2,4);hold off, cla;
for t = 1:N
    v = spm_unvec(DEM.qU.a{2}(:,t),action);

    for i = 1:n
        pos = v.position(:,i);
        col = v.signal(end - 2:end,i);
        col = max(min(col,1),0);
        plot(pos(2),pos(1),'.','markersize',8,'color',full(col)); hold on

        % destination
        %------------------------------------------------------------------
        if t == N
            plot(pos(2),pos(1),'.','markersize',16,'color',full(col));   hold on
            plot(pos(2),pos(1),'h','markersize',12,'color',full(h*col)); hold on
        end
    end
    set(gca,'Color','k');
    axis square, box off
    axis([-1 1 -1 1]*A)
    drawnow
    
    % save
    %----------------------------------------------------------------------
    Mov(t) = getframe(gca);
    
end
 
set(gca,'Userdata',{Mov,8})
set(gca,'ButtonDownFcn','spm_DEM_ButtonDownFcn')
title('Extrinsic','FontSize',16)
xlabel('location')

% save movie frames as images (Octave-compatible)
%--------------------------------------------------------------------------
movieDir = 'morphogenesis_frames';
if ~exist(movieDir, 'dir')
    mkdir(movieDir);
end
for i = 1:length(Mov)
    imwrite(Mov(i).cdata, fullfile(movieDir, sprintf('frame_%03d.png', i)));
end
fprintf('Frames saved to %s/ (use ffmpeg to create video)\n', movieDir);

% dump schema (deduped) to JSON -------------------------------------------
trace_dump_schema('dem_morphogenesis_schema.json');

end  % main function DEM_morphogenesis


% Equations of motion and observer functions
%==========================================================================
 
% sensed signal
%--------------------------------------------------------------------------
function sense = morphogenesis(position, signal, y)
% x - position of cells
% s - signals released
% y - location of sampling [default: x]
%__________________________________________________________________________
 
% preliminaries
%--------------------------------------------------------------------------
if nargin < 3; y = position; end           % sample positions

trace_sig('morphogenesis', ...
    {'position', position; 'signal', signal; 'y', y}, ...
    {});  % output logged at end

n     = size(y, 2);                        % number of locations
m     = size(signal, 1);                   % number of signals
decay = 1;                                 % signal decay over space 
sense = zeros(m,n);                        % signal sensed at each location
for i = 1:n
    for j = 1:size(position, 2)
        
        % distance
        %------------------------------------------------------------------
        distance = y(:,i) - position(:,j);
        distance = sqrt(distance'*distance);
        
        % signal concentration
        %------------------------------------------------------------------
        sense(:,i) = sense(:,i) + exp(-decay*distance).*signal(:,j);

    end
end

trace_sig('morphogenesis', ...
    {'position', position; 'signal', signal; 'y', y}, ...
    {'sense', sense});
end


% CellState: the observable state of each cell
%--------------------------------------------------------------------------
function state = CellState(position, signal, sense)
    state.position = position;
    state.signal = signal;
    state.sense = sense;
end


% observe: generative process - maps actions to actual observations
%--------------------------------------------------------------------------
function observed = observe(_position, v, action, P)
global t
if isempty(t);
    signalStrength = 0;
else
    signalStrength = (1 - exp(-t*2));
end

trace_sig('observe', ...
    {'_position', _position; 'v', v; 'action', action; 'P', P}, ...
    {});
trace_sig('spm_unvec', ...
    {'action_vec', action; 'template', P}, ...
    {});

action = spm_unvec(action, P);

trace_sig('spm_unvec', ...
    {'action_vec', action; 'template', P}, ...
    {'action_struct', action});

position = action.position;
signal = action.signal;

trace_sig('morphogenesis', ...
    {'position', action.position; 'signal', action.signal}, ...
    {});
sense_raw = morphogenesis(action.position, action.signal);
trace_sig('morphogenesis', ...
    {'position', action.position; 'signal', action.signal}, ...
    {'sense', sense_raw});
sense = signalStrength * sense_raw;

trace_sig('observe', ...
    {'_position', _position; 'v', v; 'action', action; 'P', P}, ...
    {'observed', CellState(position, signal, sense)});

observed = CellState(position, signal, sense);
end


% expect: generative model - maps beliefs to expected observations
%--------------------------------------------------------------------------
function expected = expect(_position, identityLogits, P)
global t
if isempty(t);
    signalStrength = 0;
else
    signalStrength = (1 - exp(-t*2));
end

trace_sig('expect', ...
    {'_position', _position; 'identityLogits', identityLogits; 'P', P}, ...
    {});
trace_sig('spm_softmax', ...
    {'logits', identityLogits}, ...
    {});

identityBelief = spm_softmax(identityLogits);

trace_sig('spm_softmax', ...
    {'logits', identityLogits}, ...
    {'belief', identityBelief});

position = P.position * identityBelief;
signal = P.signal * identityBelief;
sense = signalStrength * P.sense * identityBelief;

trace_sig('expect', ...
    {'_position', _position; 'identityLogits', identityLogits; 'P', P}, ...
    {'expected', CellState(position, signal, sense)});

expected = CellState(position, signal, sense);
end


function trace_sig(node_name, inputs_kv, outputs_kv)
% Deduplicated signature logger for schema discovery.
% inputs_kv / outputs_kv: N-by-2 cell array: {name, value; ...}
global TRACE_SCHEMA TRACE_SCHEMA_KEYS

in_s  = trace_describe_kv(inputs_kv);
out_s = trace_describe_kv(outputs_kv);

key = trace_make_key(node_name, in_s, out_s);
field = trace_key_to_field(key);

if isfield(TRACE_SCHEMA_KEYS, field)
    return;
end
TRACE_SCHEMA_KEYS.(field) = true;

event = struct();
event.name   = node_name;
event.inputs = in_s;
event.outputs = out_s;
TRACE_SCHEMA{end+1} = event;
end


function desc = trace_describe_kv(kv)
% Convert {name,value} list into array of {name,class,size}
if isempty(kv)
    desc = {};
    return;
end

desc = cell(size(kv,1), 1);
for i = 1:size(kv,1)
    nm = kv{i,1};
    v  = kv{i,2};
    d  = trace_describe_value(v);
    d.name = nm;
    desc{i} = d;
end
end


function d = trace_describe_value(v)
d = struct();
d.class = class(v);

% size: always a row vector (JSON-friendly)
try
    sz = size(v);
catch
    sz = [];
end
d.size = sz(:)';  % force row

% if struct: capture fields as part of "class" effectively (still minimal)
if isstruct(v)
    f = fieldnames(v);
    % Keep minimal but stable: encode fieldnames into class-like label
    d.class = ['struct{' strjoin(f(:)', ',') '}'];
end
end


function key = trace_make_key(node_name, in_s, out_s)
key = [node_name '|in:' trace_pack_desc(in_s) '|out:' trace_pack_desc(out_s)];
end


function s = trace_pack_desc(desc_cell)
% Serialize descriptor cells to a stable string
if isempty(desc_cell)
    s = '[]';
    return;
end

parts = cell(numel(desc_cell), 1);
for i = 1:numel(desc_cell)
    d = desc_cell{i};
    nm = d.name;
    cl = d.class;
    sz = d.size;
    szs = sprintf('%dx', sz);
    if ~isempty(szs)
        szs = szs(1:end-1); % drop trailing x
    else
        szs = '';
    end
    parts{i} = [nm ':' cl ':' szs];
end
s = strjoin(parts, ';');
end


function field = trace_key_to_field(key)
% Make a safe struct field name from arbitrary key
field = regexprep(key, '[^a-zA-Z0-9_]', '_');
if isempty(field) || ~isletter(field(1))
    field = ['k_' field];
end
end


function trace_dump_schema(out_path)
global TRACE_SCHEMA
if isempty(TRACE_SCHEMA)
    return;
end

payload = struct();
payload.generated_at = datestr(now(), 30);
payload.events = {TRACE_SCHEMA{:}};  % cell -> cell of structs

json = trace_jsonencode(payload);

fid = fopen(out_path, 'w');
if fid < 0
    error('Could not open schema output path: %s', out_path);
end
fwrite(fid, json);
fwrite(fid, sprintf('\n'));
fclose(fid);

fprintf('Schema saved to %s\n', out_path);
end


function json = trace_jsonencode(x)
% Use jsonencode if present; otherwise minimal fallback encoder.
if exist('jsonencode', 'builtin') || exist('jsonencode', 'file')
    json = jsonencode(x);
else
    json = trace_json_fallback(x);
end
end


function s = trace_json_fallback(x)
% Minimal JSON encoder for structs/cells/chars/numeric/logical.
if isstruct(x)
    f = fieldnames(x);
    parts = cell(numel(f),1);
    for i = 1:numel(f)
        k = f{i};
        parts{i} = [trace_json_string(k) ':' trace_json_fallback(x.(k))];
    end
    s = ['{' strjoin(parts, ',') '}'];
elseif iscell(x)
    parts = cell(numel(x),1);
    for i = 1:numel(x)
        parts{i} = trace_json_fallback(x{i});
    end
    s = ['[' strjoin(parts, ',') ']'];
elseif ischar(x)
    s = trace_json_string(x);
elseif isnumeric(x)
    if isempty(x)
        s = '[]';
    elseif isscalar(x)
        s = trace_json_number(x);
    else
        % encode numeric arrays as nested arrays up to 2D; otherwise flatten
        nd = ndims(x);
        if nd > 2
            x = x(:);
        end
        if isvector(x)
            parts = arrayfun(@(v) trace_json_number(v), x(:)', 'UniformOutput', false);
            s = ['[' strjoin(parts, ',') ']'];
        else
            rows = cell(size(x,1),1);
            for r = 1:size(x,1)
                parts = arrayfun(@(v) trace_json_number(v), x(r,:), 'UniformOutput', false);
                rows{r} = ['[' strjoin(parts, ',') ']'];
            end
            s = ['[' strjoin(rows, ',') ']'];
        end
    end
elseif islogical(x)
    if isscalar(x)
        s = ternary(x, 'true', 'false');
    else
        parts = arrayfun(@(v) ternary(v, 'true', 'false'), x(:)', 'UniformOutput', false);
        s = ['[' strjoin(parts, ',') ']'];
    end
else
    % fallback: stringify
    s = trace_json_string(['<' class(x) '>']);
end
end


function s = trace_json_string(str)
str = strrep(str, '\', '\\');
str = strrep(str, '"', '\"');
str = strrep(str, sprintf('\n'), '\n');
str = strrep(str, sprintf('\r'), '\r');
str = strrep(str, sprintf('\t'), '\t');
s = ['"' str '"'];
end


function s = trace_json_number(v)
if isnan(v)
    s = 'null';
elseif isinf(v)
    s = 'null';
else
    s = sprintf('%.17g', v);
end
end


function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
