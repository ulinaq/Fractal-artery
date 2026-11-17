% function V = TracerForward(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,vsegment,x)
% Compute tracer indicator in the vascular system.
%
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
% Aa = Flow(P,Grid,Trans.ay,Trans.ax);
% Av = Flow(Pv,Grid,Trans.vy,Trans.vx);

%% Matrix volumetric flow rate between parts of segmented vessel
Co = node.n + vnode.n;  Cv = Co + segment.n; Ca = length(node.input); Cq = Cv+vsegment.n;
NAs = sum(cellfun('length',segment.Cnum));  %total number of segment parts in the arterial network
AT = sparse(node.n,node.n); %Arterial network node
ATS = sparse(node.n,NAs);   %Connection node -> segment
AST = sparse(NAs,node.n);   %Connection segment -> node
AS = sparse(NAs,NAs);       %Arterial segments
NVs = sum(cellfun('length',vsegment.ind));    % total number of segments in vein network
VT = sparse(vnode.n,vnode.n);   %Vein nodes
VTS = sparse(vnode.n,NVs);      %Venous node->segments
VST = sparse(NVs,vnode.n);      %segment->node
VS = sparse(NVs,NVs);           %venous segments
NCs = sum(cellfun('length',Connect.ind)); % # of segment parts in direct connection
AC = sparse(NCs,NCs);           %dirCon segment parts
ATC = sparse(node.n,NCs);       %artery node-> dircon
ACT = sparse(NCs,node.n);       %dircon -> artery node
VTC = sparse(vnode.n,NCs);      %vein node-> dircon
VCT = sparse(NCs,vnode.n);      %dircon -> vein node
vol = zeros(Co+NCs+NAs+NVs,1);  %volume of discrete parts (of nodes and vessel segments)
% volume in nodes
for i = 1:node.n
    vol(i) = numel(node.ind{i})*Grid.volume;
end
for i = 1:Connect.n
    vol(node.n+Connect.Cnum{i}) = (pi*Connect.rad(i)^2*Grid.hx);   
end
for i = 1:vnode.n
    vol(node.n+NCs+i) = numel(vnode.ind{i})*Grid.volume;
end
%% Arterial tree matrix
for i = 1 : segment.n
    Fl = x(Co+i);%/(pi*segment.rad(i)^2*Grid.hx);%
    vol(Co+NCs+segment.Cnum{i}) = (pi*segment.rad(i)^2*Grid.hx);
    ns = length(segment.Cnum{i});
%     if Fl > sum(x(Co+segment.isroot))*1e-4
    if Fl>0
        AT(segment.init(i),segment.init(i)) = AT(segment.init(i),segment.init(i))-Fl;
        AST(segment.Cnum{i}(end),segment.init(i)) = Fl;
        AS(segment.Cnum{i},segment.Cnum{i}) = Fl*(-speye(ns)+spdiags(ones(ns,1),1,ns,ns));
        ATS(segment.end(i),segment.Cnum{i}(1)) = Fl;
    else
        AT(segment.end(i),segment.end(i)) = AT(segment.end(i),segment.end(i))+Fl;
        AST(segment.Cnum{i}(1),segment.end(i)) = -Fl;
        AS(segment.Cnum{i},segment.Cnum{i}) = -Fl*(-speye(ns)+spdiags(ones(ns,1),-1,ns,ns));
        ATS(segment.init(i),segment.Cnum{i}(end)) = -Fl;
%     end
    end
end

%% Venal tree
for i = 1 : vsegment.n
    Fl = x(Cv+i);%/(pi*vsegment.rad(i)^2*Grid.hx);%*slow;%
    vol(Co+NCs+NAs+vsegment.Cnum{i}) = (pi*vsegment.rad(i)^2*Grid.hx);
    ns = length(vsegment.Cnum{i});
%     if Fl > sum(x(Cv+vsegment.isroot))*1e-4
    if Fl<0
        VT(vsegment.init(i),vsegment.init(i)) = VT(vsegment.init(i),vsegment.init(i))+Fl;
        VST(vsegment.Cnum{i}(end),vsegment.init(i)) = -Fl;
        VS(vsegment.Cnum{i},vsegment.Cnum{i}) = -Fl*(-speye(ns)+spdiags(ones(ns,1),1,ns,ns));
        VTS(vsegment.end(i),vsegment.Cnum{i}(1)) = -Fl;
    else
        VT(vsegment.end(i),vsegment.end(i)) = VT(vsegment.end(i),vsegment.end(i))-Fl;
        VST(vsegment.Cnum{i}(1),vsegment.end(i)) = Fl;
        VS(vsegment.Cnum{i},vsegment.Cnum{i}) = Fl*(-speye(ns)+spdiags(ones(ns,1),-1,ns,ns));
        VTS(vsegment.init(i),vsegment.Cnum{i}(end)) = Fl;
%     end
    end
    if vsegment.isroot(i)
        VT(vsegment.init(i),vsegment.init(i)) = -abs(Fl);
    end
end

%% Coupling between arterial terminal nodes and venous terminal nodes (direct connection)
for i = 1 : Connect.n
    Fl = x(Cq+i);%/(pi*Connect.rad(i)^2*Grid.hx);%*slow;
    ns = length(Connect.Cnum{i});
    if Fl>0
        AT(Connect.AV(i,1),Connect.AV(i,1)) = AT(Connect.AV(i,1),Connect.AV(i,1))-Fl;    % arterial terminal
        ACT(Connect.Cnum{i}(1),Connect.AV(i,1)) = Fl;
        AC(Connect.Cnum{i},Connect.Cnum{i}) = Fl*(-speye(ns)+spdiags(ones(ns,1),-1,ns,ns));
        VTC(Connect.AV(i,2),Connect.Cnum{i}(end)) = Fl; 
    else
        VT(Connect.AV(i,2),Connect.AV(i,2)) = VT(Connect.AV(i,2),Connect.AV(i,2))+Fl;   %vein terminal connection
        VCT(Connect.Cnum{i}(end),Connect.AV(i,2)) = -Fl;
        AC(Connect.Cnum{i},Connect.Cnum{i}) = -Fl*(-speye(ns)+spdiags(ones(ns,1),1,ns,ns));
        ATC(Connect.AV(i,1),Connect.Cnum{i}(1)) = -Fl;
    end    
end

%% Tracer concentration iteration
tic;
Tt = 2;
dt = 0.000005;
TL = 0 : dt : Tt;
Tc = numel(TL);
c = zeros(Co+NCs+NAs+NVs,1);
% Initial concentration in arterial root
C0 = 1;
t0 = 0.1;
%fixed parameters
a  = 2;   b  = 1;
%change from minutes to seconds
t  = TL*60; t0 = t0*60; dt=dt*60;
dV = round(0.25/dt); % video data iteration interval
dn = 20;    % Coarsening factor
%get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
AIF = AIF.*(t>=t0);
%% Combine block matrices
I = spdiags(vol); 
A = [AT,ATC,sparse(node.n,vnode.n),ATS,sparse(node.n,NVs);...
    ACT,AC,VCT,sparse(NCs,NAs+NVs);...
    zeros(vnode.n,node.n),VTC,VT,sparse(vnode.n,NAs),VTS;...
    AST,sparse(NAs,NCs+vnode.n),AS,sparse(NAs,NVs);...
    sparse(NVs,NCs+node.n),VST,sparse(NVs,NAs),VS];
% A = speye(Co+NCs+NAs+NVs) + dt/Grid.volume*[AT,ATC,sparse(node.n,vnode.n),ATS,sparse(node.n,NVs);...
%     ACT,AC,VCT,sparse(NCs,NAs+NVs);...
%     zeros(vnode.n,node.n),VTC,VT,sparse(vnode.n,NAs),VTS;...
%     AST,sparse(NAs,NCs+vnode.n),AS,sparse(NAs,NVs);...
%     sparse(NVs,NCs+node.n),VST,sparse(NVs,NAs),VS];
% Color scale
clim = [0,1e-3*max(AIF)];
% Open movie
folder = ['Result/',date,'c/'];
[~,~,~] = mkdir(folder);
% subfolder = [folder,'/A';folder,'/V';folder,'/C';folder,'/I';folder,'/L'];
% for i =1:5
%     [~,~,~] = mkdir(subfolder(i,:));
% end
pathsave = [folder '/Coarse3.avi'];
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 50;
open(v);
H = figure(3);
% iteration
lm = 1;  % Counter to save the data
csum = c;
coSave = sparse(Grid.N,round(Tc/dV)-1);
cSave = zeros(Co+NCs+NAs+NVs,round(Tc/dV)-1);
cISave = zeros(Co+NCs+NAs+NVs,round(Tc/dV)-1);
% cSave = sparse(round(Grid.Nx/dn)*round(Grid.Ny/dn),round(Tc/dV));
%waitbar
f = waitbar(0,'1','Name','Computing Image Contrast',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(f,'canceling',0);
time = [];
%% Compute vascular coverage volume in every segmentation pixel, 
% each vessel has value 1 in pixel, an overlapping pixel will have value
% more than 1, tissues are zeros.
cVol = zeros(Grid.Nx,Grid.Ny); 
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cVol(segment.distrind{i}{j})=cVol(segment.distrind{i}{j})+1;%/numel(segment.distrind{i}{j});
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cVol(vsegment.distrind{i}{j})=cVol(vsegment.distrind{i}{j})+1;%/numel(vsegment.distrind{i}{j});
    end
end
for i=1:node.n
    cVol(node.ind{i})=cVol(node.ind{i})+1;%/numel(node.ind{i});
end
for i=1:vnode.n
    cVol(vnode.ind{i})=cVol(vnode.ind{i})+1;%/numel(vnode.ind{i});
end
for i=1:Connect.n
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    cVol(Connect.distrind{i}{j})=cVol(Connect.distrind{i}{j})+1;
    end
end
cVol(cVol==0) = 1;
cVol = cVol*Grid.volume;
% cVol(cVol==0) = 1;      % set tissue as 1 to avoid division by 0;

%% Main iteration tracer flow
bc = zeros(Co+NCs+NAs+NVs,1);
for iter = 1 : Tc  
bc(1:2) = AIF(iter);
c = max(c.*vol+dt*A*(c+bc),0)./vol;
csum = csum+c;
% if iter==round(Tc/2)
%     dV = 20*dV;
% end
if mod(iter,dV)==0 %&& iter>0.001*Tc
%     % Check for clicked Cancel button
    if getappdata(f,'canceling')
        break
        delete(f);
%         close(v);
    end
    time = [time;t(iter)];
%     % Update waitbar and message
    waitbar(iter/numel(TL),f,sprintf([num2str(iter/numel(TL)*100),'  ( ',num2str(toc),' s )']))
    cInt = csum./dV;
%     cInt=c;
% Mapping concentration into image: number of molecule are summed up, 
% Cmap = ca*Va/#pixel + cv*Vv/#pixel + cc*Vc/#pixel
    cA = zeros(Grid.Nx,Grid.Ny); cV = zeros(Grid.Nx,Grid.Ny);  cO = zeros(Grid.Nx,Grid.Ny);
for i=1:segment.n
    Vols = (pi*segment.rad(i)^2*Grid.hx);
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cA(segment.distrind{i}{j})=cA(segment.distrind{i}{j})+cInt(Co+NCs+segment.Cnum{i}(j))*Vols/numel(segment.distrind{i}{j});
    end
end
for i=1:vsegment.n
    Vols = (pi*vsegment.rad(i)^2*Grid.hx);
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=cV(vsegment.distrind{i}{j})+cInt(Co+NCs+NAs+vsegment.Cnum{i}(j))*Vols/numel(vsegment.distrind{i}{j});
    end
end
for i=1:node.n
    cA(node.ind{i})=cA(node.ind{i})+cInt(i)*Grid.volume;%/numel(node.ind{i});
end
for i=1:vnode.n
    cV(vnode.ind{i})=cV(vnode.ind{i})+cInt(node.n+NCs+i)*Grid.volume;%/numel(vnode.ind{i});
end
for i=1:Connect.n
    Vols = (pi*Connect.rad(i)^2*Grid.hx);
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    cO(Connect.distrind{i}{j})=cO(Connect.distrind{i}{j})+cInt(node.n+Connect.Cnum{i}(j))*Vols/numel(Connect.distrind{i}{j});
    end
end
Img = (cA+cV+cO)/Grid.volume;
figure(H);
imagesc(Img,clim*Grid.volume);colormap(bone);axis image;axis off; title('Direct Connection');daspect([1 1 1])
frame = getframe(gcf);
writeVideo(v,frame);
coSave(:,lm)=Img(:);
cSave(:,lm)=cInt;
cISave(:,lm)=csum./dV;
csum = 0;
lm=lm+1;
end
end
close(v);
delete(f);
toc

function cdot = fCon(t, c, A)
% Initial concentration in arterial root
C0 = 1;
t0 = 0.1;
%fixed parameters
a  = 2;   b  = 1;
%get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
AIF = AIF.*(t>=t0);

bc = zeros(Co+NCs+NAs+NVs,1);
bc(1:2) = AIF(iter);
cdot = A*(c+bc)./vol;
end

%% Functions


% Dormand-prince method
function [t, y] = dormandprince(odefun, ...   % The ODE function
                                y0, ...       % Initial states
                                ts, ...       % Start time
                                tf, ...       % End time
                                h, ...        % Step time
                                TOL)          % Local error tollerance
y(:,1) = y0;

k = 2;
t = ts;
while t < tf
    yn = y(:,k-1);
    tn = t(k-1);
    k1 = h * odefun(tn, yn);
    k2 = h * odefun(tn+h/5, yn+k1/5);
    k3 = h * odefun(tn+h*3/10, yn+k1*3/40 + k2*9/40);
    k4 = h * odefun(tn+h*4/5, yn + k1*44/45 - k2*56/15 + k3*32/9);
    k5 = h * odefun(tn+h*8/9, yn + k1*19372/6561 - k2*25360/2187 + ...
                    k3*64448/6561 - k4*212/729);
    k6 = h * odefun(tn+h, yn + k1*9017/3168 - k2*355/33 + ...
                    k3*46732/5247 + k4*49/176 - k5*5103/18656);
    k7 = h * odefun(tn+h, yn + k1*35/384 + k3*500/1113 + k4*125/192 - ...
                    k5*2187/6784 + k6*11/84);

    Y = yn + 35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + ...
        11/84*k6;
    Z = yn + 5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - ...
        92097/339200*k5 + 187/2100*k6 + 1/40*k7;

    E = norm(Y-Z); % local error estimation
    h = 0.9 * h * (TOL/E)^(1/5);

    y(:,k) = Y;
    t(k) = t(k-1) + h;
    k = k + 1;
end
end

% Runge-Kutta-Fehlberg method (RKF45)
% This is an adaptive method with variable time step.
function [t, y] = rkf45(odefun, ...   % The ODE function
                        y0, ...       % Initial states
                        ts, ...       % Start time
                        tf, ...       % End time
                        h, ...        % Step time
                        TOL)          % Local error tollerance
y(:,1) = y0;

k = 2;
t = ts;
while t < tf
    yn = y(:,k-1);
    tn = t(k-1);
    k1 = h * odefun(tn, yn);
    k2 = h * odefun(tn+h/4, yn+k1/4);
    k3 = h * odefun(tn+h*3/8, yn+k1*3/32 + k2*9/32);
    k4 = h * odefun(tn+h*12/13, yn + k1*1932/2197 - k2*7200/2197 + ...
                    k3*7296/2197);
    k5 = h * odefun(tn+h, yn + k1*439/216 - 8*k2 + k3*3680/513 - ...
                    k4*845/4104);
    k6 = h * odefun(tn+h/2, yn - k1*8/27 + k2*2 - k3*3544/2565 + ...
                    k4*1859/4104 - k5*11/40);

    Y = yn + 25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5;
    Z = yn + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + ...
        2/55*k6;

    E = norm(Y-Z); % local error estimation
    h = 0.9 * h * (TOL/E)^(1/5);

    y(:,k) = Y;
    t(k) = t(k-1) + h;
    k = k + 1;
end
end

% Backwards Euler Method
function [t, y] = beuler(odefun, ... % The ODE function
                         y0, ...     % Initial states
                         ts, ...     % Start time
                         tf, ...     % End time
                         h, ...      % Step time
                         TOL, ...    % Tollerance for Newton's method
                         MAX_ITER)   % Max iterations for Newton's method

if nargin < 6
    TOL = 0.0001;
    MAX_ITER = 1000;
elseif nargin < 7
    MAX_ITER = 1000;
end

t = ts:h:tf;
y = zeros(length(y0),length(t));
I = eye(length(y0));

y(:,1) = y0;

for k = 1 : length(t)-1
    y1 = y(:,k);
    F = build_F(odefun, t(k), y(:,k), y1, h);
    y2 = y1 - (I-h*Jacobian(odefun, t(k), y1)) \ F;

    n_iterations = 1;
    while (norm(y2-y1) > TOL && n_iterations < MAX_ITER)
        y1 = y2;
        F = build_F(odefun, t(k), y(:,k), y1, h);
        y2 = y1 - (I-h*Jacobian(odefun, t(k), y1)) \ F;
        n_iterations = n_iterations + 1;
    end

    y(:,k+1) = y2;
end
end

% -------------------------------------------------------------------------
% Build the equation: F = 0
% -------------------------------------------------------------------------
function F = build_F(f, t, y1, y2, h)
F = y2 - y1 - h*f(t, y2);
end

% -------------------------------------------------------------------------
% Calculate Jacobian of function f at given x
% Complex-Step Derivative Approximation (CSDA) Method
% See: https://github.com/auralius/numerical-jacobian
% -------------------------------------------------------------------------
function DF = Jacobian(f, t, x, epsilon)
if nargin < 4
    epsilon = 1e-5;
end

epsilon_inv = 1/epsilon;

nx = length(x); % Dimension of the input x;

% Do perturbation
for k = 1 : nx
    xplus = x;
    xplus(k) =  x(k) + 1i*epsilon;
    DF(:, k) = imag(f(t, xplus))  .* epsilon_inv;
end
end

function An = Flow(Pa,Grid,Trany,Tranx)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
   V.y = zeros(Nx,Ny+1);
   V.x = zeros(Nx+1,Ny);
   V.y(:,2:Ny) = (Pa(:,1:Ny-1)-Pa(:,2:Ny)).*Trany(:,2:Ny);
   V.x(2:Nx,:) = (Pa(1:Nx-1,:)-Pa(2:Nx,:)).*Tranx(2:Nx,:);

   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.u = sqrt(V.qx.^2+V.qy.^2);
   u = reshape(V.u,N,1);
   V.range = max(u)-min(u);
   % define neighbor cell
num = reshape([1:N],Nx,Ny);
Yin = [zeros(Nx,1) num];
Yout = [num zeros(Nx,1)];
Xin = [zeros(1,Ny); num];
Xout = [num; zeros(1,Ny)];
face.neighbors = [reshape(Xin,[],1), reshape(Xout,[],1);reshape(Yin,[],1), reshape(Yout,[],1)]; % face area between 2 cells
face.flux = [reshape(V.x,[],1);reshape(V.y,[],1)]; % flux corresponding to face 
ind = ~any(face.neighbors==0,2);
n  = double(face.neighbors(ind,:));
in = min(face.flux(ind), 0); % negative value, inflow to cell i
out  = max(face.flux(ind), 0); % positive, goes outward

% Cell wise total outflow
outflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);
An.n = n; An.in = in; An.out = out; An.outflow = outflow;
end
% end
