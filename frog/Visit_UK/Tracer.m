% function V = Tracer(prm,Pa,Pv,Grid,Trans,q,node,vnode)
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
Aa = Flow(Pa,Grid,Trans.ay,Trans.ax);
Av = Flow(Pv,Grid,Trans.vy,Trans.vx);
TC.a = zeros(Nx,Ny); TC.v = zeros(Nx,Ny);

%% Create matrix Arterial compartment

qq=[];
cell=[];
perf = prm.alpha*(Pa-Pv);
for i=1:length(node.input)
    qq=[qq; Grid.src.rate{i}*q(i)];
    cell = [cell;Grid.src.cell(:,i)];
end

qa = sparse(cell,1, qq, N,1);
% diagonal matrix = outflow from cells, including perfusion to venal compartment
outflow  = accumarray([n(:, 1); n(:, 2)], [-in; out]);
da = Aa.outflow - max(perf(:),0); % outflow to neighbors plus perfusion
% off-diagonal matrix is inflow from neighbors  
AA = sparse(Aa.n(:,2), Aa.n(:,1),  Aa.in, N, N)...
   + sparse(Aa.n(:,1), Aa.n(:,2), -Aa.out, N, N);
AA = -AA + spdiags(da, 0, N, N);

qq=[];
cell=[];
for i=length(node.input)+1:length(node.input)+length(vnode.output)
    qq=[qq; Grid.src.rate{i}*q(i)];
    cell = [cell;Grid.src.cell(:,i)];
end
qv = sparse(cell,1, qq, N,1);
% diagonal 
dv = Av.outflow + max(qv,0) + max(perf(:),0);
pv = reshape(Grid.volume*Grid.poro,N,1)/4;
[dv,pv,Av] = maxtof(pv,max(qa,0),Av,dv);
% Create matrix vein
AV  = sparse(Av.n(:,2), Av.n(:,1),  Av.in, N, N)...
   + sparse(Av.n(:,1), Av.n(:,2), -Av.out, N, N);
AV = -AV + spdiags(dv, 0, N, N);

A = [AA,spdiags(min(perf(:),0),0, N, N);spdiags(-1*max(perf(:),0),0, N, N),AV];
pp =[pa;pv];
% Solve
V.Ti = A\pp;

%% Colapsing compartments
d = Aa.outflow + Av.outflow + max(qa,0) + max(qv,0);
p = reshape(Grid.volume*Grid.poro,N,1);

% Create matrix vein
AN  = AV + AA;
Ti = AN\p;
Ti = reshape(Ti,Grid.Nx,Grid.Ny);

Ta = reshape(V.Ti(1:Grid.N),Grid.Nx,Grid.Ny);
Tv = reshape(V.Ti(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
V.Ta = Ta;  V.Tv = Tv;


% function [da,pa,Aa] = maxtof(pa,qa,Aa,da)
% % Handling of T -> inf (Boundary region included) :
% maxTOF = 50*sum(pa)/sum(qa);
% % Find cells that reach max TOF locally
% maxIn = max(da);
% aboveMax = full(pa./ (max(da, eps*maxIn)) ) > maxTOF;
% % Set aboveMax-cells to maxTOF
% da(aboveMax)  = maxIn;
% pa(aboveMax) = maxIn*maxTOF;
% % remove upstream connections
% Aa.in(aboveMax(Aa.n(:,2)))  = 0;
% Aa.out(aboveMax(Aa.n(:,1))) = 0;
% end
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
out = min(face.flux(ind), 0); % negative value, goes downright
in  = max(face.flux(ind), 0); % positive, goes upright

% Cell wise total outflow
outflow  = accumarray([n(:, 1); n(:, 2)], [-in; out]);
An.n = n; An.in = in; An.out = out; An.outflow = outflow;
end
% end
