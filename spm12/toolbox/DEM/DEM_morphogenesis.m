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
v     = randn(n,n)/8;                     % states (identity)
g     = Mg([],v,P);
action.position = g.position;                  % action (chemotaxis)
action.signal   = g.signal;                    % action (signal release)
 
 
% generative process 
%==========================================================================
R     = spm_cat({kron(eye(n,n),ones(2,2)) []; [] kron(eye(n,n),ones(4,4));
                 kron(eye(n,n),ones(4,2)) kron(eye(n,n),ones(4,4))});

% level 1 of generative process
%--------------------------------------------------------------------------
G(1).g  = @(position, v, action,P) Gg(position, v, action, P);
G(1).v  = Gg([], [], action, action);
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
M(1).g  = @(position, v, P) Mg([], v, P);
M(1).v  = g;
M(1).V  = exp(3);
M(1).pE = P;
 
% level 2: 
%--------------------------------------------------------------------------
M(2).v  = v;
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
    plot(x(2,i),x(1,i),'.','markersize',32,'color',col); hold on
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
    v = spm_unvec(DEM.qU.a{2}(:,t),a);
    for i = 1:n
        position = v.x(1,i);
        sense = v.s(end - 2:end,i);
        sense = full(max(min(sense,1),0));
        plot(t,position,'.','markersize',16,'color',sense); hold on
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
    v = spm_unvec(DEM.qU.a{2}(:,t),a);
    
    for i = 1:n
        position = v.x(:,i);
        sense = v.s(end - 2:end,i);
        sense = max(min(sense,1),0);
        plot(position(2),position(1),'.','markersize',8,'color',full(sense)); hold on
        
        % destination
        %------------------------------------------------------------------
        if t == N
            plot(position(2),position(1),'.','markersize',16,'color',full(c));   hold on
            plot(position(2),position(1),'h','markersize',12,'color',full(h*c)); hold on
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
    imwrite(Mov(i).sensedata, fullfile(movieDir, sprintf('frame_%03d.png', i)));
end
fprintf('Frames saved to %s/ (use ffmpeg to create video)\n', movieDir);

return
 
 
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
 
 
% first level process: generating input
%--------------------------------------------------------------------------
function g = Gg(_position, v, action, P)
global t
if isempty(t);
    signal = 0;
else
    signal = (1 - exp(-t*2));
end
action = spm_unvec(action, P);

g.position(1,:) = action.position(1,:);                   % position  signal
g.position(2,:) = action.position(2,:);                   % position  signal
g.signal = action.signal;                                 % intrinsic signal
g.sense      = signal * morphogenesis(action.position, action.signal);  % extrinsic signal


% first level model: mapping hidden causes to sensations
%--------------------------------------------------------------------------
function g = Mg(_position, v, P)
global t
if isempty(t);
    signal = 0;
else
    signal = (1 - exp(-t*2));
end

p = spm_softmax(v);                   % expected identity

g.position = P.position*p;            % position
g.signal   = P.signal*p;              % intrinsic signal
g.sense    = signal*P.sense*p;            % extrinsic signal