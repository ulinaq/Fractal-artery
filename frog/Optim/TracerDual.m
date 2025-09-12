% function V = TracerForward(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,vsegment,x)
% Compute tracer indicator in the vascular system.
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
Aa = Flow(Pa,Grid,Trans.ay,Trans.ax);
Av = Flow(Pv,Grid,Trans.vy,Trans.vx);
perf = prm.alpha*(Pa-Pv);

%% Arterial tree matrix
Co = node.n + vnode.n;  Cv = Co + segment.n; Ca = length(node.input);
NAs = sum(cellfun('length',segment.Cnum));
AT = sparse(node.n,node.n);
ATS = sparse(node.n,NAs);
AST = sparse(NAs,node.n);
AS = sparse(NAs,NAs);
for i = 1 : segment.n
    Fl = x(Co+i);%/(pi*segment.len(i)*segment.rad(i)^2);
    ns = length(segment.Cnum{i});
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
    end    
end

%% Venal tree
NVs = sum(cellfun('length',vsegment.c));
VT = sparse(vnode.n,vnode.n);
VTS = sparse(vnode.n,NVs);
VST = sparse(NVs,vnode.n);
VS = sparse(NVs,NVs);
for i = 1 : vsegment.n
    Fl = x(Cv+i);%/(pi*vsegment.len(i)*vsegment.rad(i)^2);
    ns = length(vsegment.Cnum{i});
    if Fl<0
        VT(vsegment.init(i),vsegment.init(i)) = VT(vsegment.init(i),vsegment.init(i))+Fl;
        VST(vsegment.Cnum{i}(end),vsegment.init(i)) = -Fl;
        VS(vsegment.Cnum{i},vsegment.Cnum{i}) = -Fl*(-speye(ns)+spdiags(ones(ns,1),-1,ns,ns));
        VTS(vsegment.end(i),vsegment.Cnum{i}(1)) = -Fl;
    else
        VT(vsegment.end(i),vsegment.end(i)) = VT(vsegment.end(i),vsegment.end(i))-Fl;
        VST(vsegment.Cnum{i}(1),vsegment.end(i)) = Fl;
        VS(vsegment.Cnum{i},vsegment.Cnum{i}) = Fl*(-speye(ns)+spdiags(ones(ns,1),1,ns,ns));
        VTS(vsegment.init(i),vsegment.Cnum{i}(end)) = Fl;
    end
    if vsegment.isroot(i)
        VT(vsegment.init(i),vsegment.init(i)) = -Fl;
    end
end

%% Coupling between arterial terminal nodes and capillary
Artin = node.num(node.isterminal);
qq = []; qa = []; cell = []; % temporary variable
ACT = sparse(node.n,N);
ATC = sparse(N,node.n);
Ca = length(node.input);

for i=1:Ca
    if q(i)>0 % flow into capillary
        ATC = ATC + sparse(Grid.src.cell{i},Artin(i), max(Grid.src.rate{i}*q(i),0), N,node.n);
        AT(Artin(i),Artin(i)) = - q(i);%/(pi*segment.len(Artin(i))*segment.rad(Artin(i))^2);
    else % flow from capillary to artery
        ACT = ACT + sparse(Artin(i),Grid.src.cell{i}, -1*min(Grid.src.rate{i}*q(i),0), node.n,N);      
        qq = [qq; Grid.src.rate{i}*q(i)];
        cell = [cell;Grid.src.cell{i}];
    end
end
qa = sparse(cell,1, qq, N,1);

%% Create matrix Arterial compartment
% diagonal matrix = outflow from cells, including perfusion to venal compartment
% outflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);
da = qa + Aa.outflow - max(perf(:),0); % outflow to neighbors, artery, and perfusion to venal comp
% off-diagonal matrix is inflow from neighbors 
AC = sparse(Aa.n(:,2), Aa.n(:,1),  Aa.out, N, N)...
   + sparse(Aa.n(:,1), Aa.n(:,2), -Aa.in, N, N);
AC = (AC + spdiags(da, 0, N, N));

%% Coupling between venous terminal nodes and capillary
Artin = vnode.num(vnode.isterminal);
qq = []; qv = []; cell = []; % temporary variable
VCT = sparse(vnode.n,N);
VTC = sparse(N,vnode.n);
for i=Ca+1:Ca+length(vnode.output)
    if q(i)>0
        VTC = VTC + sparse(Grid.src.cell{i},Artin(i-Ca), max(Grid.src.rate{i}*q(i),0), N,vnode.n);
        VT(Artin(i-Ca),Artin(i-Ca)) = - q(i);%/(pi*vsegment.len(Artin(i-Ca))*vsegment.rad(Artin(i-Ca))^2);
    else
        VCT = VCT + sparse(Artin(i-Ca),Grid.src.cell{i}, -1*min(Grid.src.rate{i}*q(i),0), vnode.n,N);
        qq = [qq; Grid.src.rate{i}*q(i)];
        cell = [cell;Grid.src.cell{i}];
    end
end
qv = sparse(cell,1, qq, N,1);

%% Create matrix Venal compartment
dv = qv + Av.outflow + min(perf(:),0); % outflow to neighbors, vein, and perfusion to arterial comp
% off diagonal
VC  = sparse(Av.n(:,2), Av.n(:,1),  -Av.in, N, N)...
   + sparse(Av.n(:,1), Av.n(:,2), Av.out, N, N);
VC = (VC + spdiags(dv, 0, N, N));


%% Tracer concentration iteration
Tt = 1;
dt = 0.0001;
TL = 0 : dt : Tt;
Tc = numel(TL);
c = zeros(Co+2*Grid.N+NAs+NVs,1);
% Initial concentration in arterial root
C0 = 1;
t0 = 0;
%fixed parameters
a  = 1.5;   b  = 10;
%change from minutes to seconds
t  = TL*60; t0 = t0*60; dt=dt*60;
dV = 10*round(0.05/dt); % video data iteration interval
dn = 30;    % Coarsening factor
%get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
root = node.num(node.isroot);
% AIF = AIF.*(t>=t0);
% c(root,1) = AIF(1);
%% Combine block matrices
A = speye(Co+2*N+NAs+NVs) + dt*[AT,ACT,sparse(node.n,(N+vnode.n)),ATS,sparse(node.n,NVs);...
    ATC,AC,spdiags(-1*min(perf(:),0),0, N, N),sparse(N,vnode.n+NAs+NVs);...
    sparse(N,node.n),spdiags(max(perf(:),0),0, N, N),VC,VTC,sparse(N,NAs+NVs);...
    sparse(vnode.n,(N+node.n)),VCT/prm.Por,VT,sparse(vnode.n,NAs),VTS;...
    AST,sparse(NAs,2*N+vnode.n),AS,sparse(NAs,NVs);...
    sparse(NVs,2*N+node.n),VST,sparse(NVs,NAs),VS];
% Color scale
clim = [0,0.5*max(AIF)];
% Open movie
folder = ['Result/',date,'/'];
subfolder = [folder,'/I';folder,'/L'];
for i =1:2
    [~,~,~] = mkdir(subfolder(i,:));
end
pathsave = [folder '/_Coarse.avi'];
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 100;
open(v);
H = figure('Position',[300,200,700,400]);

% iteration
lm = 1;  % Counter to save the data
csum = c;
cCombSave = sparse(Grid.N,round(Tc/dV));
cSave = sparse(round(Grid.Nx/dn)*round(Grid.Ny/dn),round(Tc/dV));

for iter = 2 : Tc
c = max(A*c,0);
c(1:2) = AIF(iter);
csum = csum+c;
if iter>numel(TL)/3
    dV = 4*dV;
end
if mod(iter,dV)==0
    cInt = csum/dV;
    cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny);
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cA(segment.distrind{i}{j})=cInt(Co+2*Grid.N+segment.Cnum{i}(j));
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=cInt(Co+2*Grid.N+NAs+vsegment.Cnum{i}(j));
    end
end
for i=1:node.n
    cA(node.ind{i})=cInt(i);
end
for i=1:vnode.n
    cV(vnode.ind{i})=cInt(node.n+2*Grid.N+i);
end
cC = sparse(reshape(cInt(node.n+1:node.n+Grid.N)+cInt(node.n+Grid.N+1:node.n+2*Grid.N),Grid.Nx,Grid.Ny));
I = Grid.B*0.5+cA+cV+cC;
I(~Grid.B) = 0;
figure(H);
subplot(2,2,1);imagesc(cA,clim);axis image;axis off;title('Arterial tree');
subplot(2,2,2);imagesc(cV,clim);axis image;axis off;title('Venous tree');
subplot(2,2,3);imagesc(cC,clim);axis image;axis off;title('Capillary');
subplot(2,2,4);imagesc(I,clim);colormap(bone);axis image;axis off;title('Combined');
frame = getframe(gcf);
writeVideo(v,frame);
cCombSave(:,lm)=I(:);
lm=lm+1;
end
end
close(v);

%% Functions
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
