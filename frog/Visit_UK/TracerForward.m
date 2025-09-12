% function V = TracerForward(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,vsegment,x)
% Compute tracer indicator in the vascular system.
%
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
% Aa = Flow(P,Grid,Trans.ay,Trans.ax);
% Av = Flow(Pv,Grid,Trans.vy,Trans.vx);

%% Arterial tree matrix
Co = node.n + vnode.n;  Cv = Co + segment.n; Ca = length(node.input); Cq = Cv+vsegment.n;
NAs = sum(cellfun('length',segment.Cnum));
AT = sparse(node.n,node.n);
ATS = sparse(node.n,NAs);
AST = sparse(NAs,node.n);
AS = sparse(NAs,NAs);
NVs = sum(cellfun('length',vsegment.c));
VT = sparse(vnode.n,vnode.n);
VTS = sparse(vnode.n,NVs);
VST = sparse(NVs,vnode.n);
VS = sparse(NVs,NVs);
NCs = sum(cellfun('length',Connect.c));
AC = sparse(NCs,NCs);
ATC = sparse(node.n,NCs);
ACT = sparse(NCs,node.n);
VTC = sparse(vnode.n,NCs);
VCT = sparse(NCs,vnode.n);
slow = 5e-2;
for i = 1 : segment.n
    Fl = x(Co+i)*slow;%/(pi*segment.rad(i)^2);
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
for i = 1 : vsegment.n
    Fl = x(Cv+i)*slow;%/(pi*vsegment.rad(i)^2);
    ns = length(vsegment.Cnum{i});
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
    end
    if vsegment.isroot(i)
        VT(vsegment.init(i),vsegment.init(i)) = -Fl;
    end
end

%% Coupling between arterial terminal nodes and capillary
for i = 1 : Connect.n
    Fl = x(Cq+i)*slow;%/(pi*Connect.rad(i)^2);
    ns = length(Connect.Cnum{i});
    if Fl>0
        ACT(Connect.Cnum{i}(1),Connect.AV(i,1)) = Fl;
        AC(Connect.Cnum{i},Connect.Cnum{i}) = Fl*(-speye(ns)+spdiags(ones(ns,1),-1,ns,ns));
        VTC(Connect.AV(i,2),Connect.Cnum{i}(end)) = Fl;
    else
        VCT(Connect.Cnum{i}(end),Connect.AV(i,2)) = -Fl;
        AC(Connect.Cnum{i},Connect.Cnum{i}) = -Fl*(-speye(ns)+spdiags(ones(ns,1),1,ns,ns));
        ATC(Connect.AV(i,1),Connect.Cnum{i}(1)) = Fl;
    end    
end

%% Tracer concentration iteration
tic;
Tt = 5;
dt = 0.00002;
TL = 0 : dt : Tt;
Tc = numel(TL);
c = zeros(Co+NCs+NAs+NVs,1);
% Initial concentration in arterial root
C0 = 1;
t0 = 0;
%fixed parameters
a  = 15;   b  = 0.15;
%change from minutes to seconds
t  = TL*60; t0 = t0*60; dt=dt*60;
dV = round(0.02/dt); % video data iteration interval
dn = 20;    % Coarsening factor
%get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
AIF = AIF.*(t>=t0);
c(1:2,1) = AIF(1);
%% Combine block matrices
A = speye(Co+NCs+NAs+NVs) + dt/Grid.volume*[AT,ATC,sparse(node.n,vnode.n),ATS,sparse(node.n,NVs);...
    ACT,AC,VCT,sparse(NCs,NAs+NVs);...
    zeros(vnode.n,node.n),VTC,VT,sparse(vnode.n,NAs),VTS;...
    AST,sparse(NAs,NCs+vnode.n),AS,sparse(NAs,NVs);...
    sparse(NVs,NCs+node.n),VST,sparse(NVs,NAs),VS];
% Color scale
clim = [0,max(AIF)];
% Open movie
folder = ['Result/',date,'c/'];
[~,~,~] = mkdir(folder);
% subfolder = [folder,'/A';folder,'/V';folder,'/C';folder,'/I';folder,'/L'];
% for i =1:5
%     [~,~,~] = mkdir(subfolder(i,:));
% end
pathsave = [folder '/Coarse.avi'];
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 100;
open(v);
H = figure();
% iteration
lm = 1;  % Counter to save the data
csum = c;
coSave = sparse(Grid.N,round(0.3252*Tc/dV));
% cSave = sparse(round(Grid.Nx/dn)*round(Grid.Ny/dn),round(Tc/dV));
%waitbar
f = waitbar(0,'1','Name','Computing Image Contrast',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(f,'canceling',0);
time = [];
for iter = 2 : Tc  
c = max(A*c,0);
c(1:2) = AIF(iter);
csum = csum+c;
% if iter==round(Tc/4)
%     dV = 10*dV;
% end
if mod(iter,dV)==0
    % Check for clicked Cancel button
    if getappdata(f,'canceling')
        break
        delete(f);
        close(v);
    end
    time = [time,t(iter)];
    % Update waitbar and message
    waitbar(iter/numel(TL),f,sprintf([num2str(iter/numel(TL)*100),'  ( ',num2str(toc),' s )']))
    cInt = csum./dV;
%     cInt=c;
    csum = 0;
    cA = zeros(Grid.Nx,Grid.Ny); cV = zeros(Grid.Nx,Grid.Ny);  cO = zeros(Grid.Nx,Grid.Ny);
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cA(segment.distrind{i}{j})=cInt(Co+NCs+segment.Cnum{i}(j));
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=cInt(Co+NCs+NAs+vsegment.Cnum{i}(j));
    end
end
for i=1:node.n
    cA(node.ind{i})=cInt(i);
end
for i=1:vnode.n
    cV(vnode.ind{i})=cInt(node.n+NCs+i);
end
for i=1:Connect.n
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    cO(Connect.distrind{i}{j})=cInt(node.n+Connect.Cnum{i}(j));
    end
end
I = Grid.B*0.005 + cA+cV+cO;
% I(~Grid.B) = 0;
figure(H);
imagesc(I,clim);colormap(bone);axis image;axis off; title('Direct Connection');daspect([1 1 1])
% subplot(2,2,1);imagesc(cA,clim);axis image;axis off;title('Arterial tree');
% subplot(2,2,2);imagesc(cV,clim);axis image;axis off;title('Venous tree');
% subplot(2,2,3);imagesc(Lores,clim);axis image;axis off;title('Coarse');
% subplot(2,2,4);
% imagesc(I,clim);colormap(bone);axis image;axis off;title('Combined');
frame = getframe(gcf);
writeVideo(v,frame);
coSave(:,lm)=I(:);
lm=lm+1;
end
end
close(v);
delete(f);
toc




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
