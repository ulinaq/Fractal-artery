function V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode)
Ny=Grid.Ny; Nx=Grid.Nx; Nz=Grid.Nz; N=Ny*Nx*Nz;
Aa = Flow(Pa,Grid,Trans.ay,Trans.ax,Trans.az);
Av = Flow(Pv,Grid,Trans.vy,Trans.vx,Trans.vz);

qq=[];
cell=[];
perf = prm.alpha*(Pa-Pv);
for i=1:length(node.input)
    qq=[qq; Grid.src.rate{i}*q(i)];
    cell = [cell;Grid.src.cell{i}];
end
qa = sparse(cell,1, qq, N,1);
% diagonal from sources
da = Aa.inflow + max(qa,0) - min(perf(:),0);
pa = reshape(Grid.volume*Grid.poro,N,1)/4;
[da,pa,Aa] = maxtof(pa,max(qa,0),Aa,da);
% Create matrix artery
AA  = sparse(Aa.n(:,2), Aa.n(:,1),  Aa.in, N, N)...
   + sparse(Aa.n(:,1), Aa.n(:,2), -Aa.out, N, N);
AA = -AA + spdiags(da, 0, N, N);

qq=[];
cell=[];
for i=length(node.input)+1:length(node.input)+length(vnode.output)
    qq=[qq; Grid.src.rate{i}*q(i)];
    cell = [cell;Grid.src.cell{i}];
end
qv = sparse(cell,1, qq, N,1);
% diagonal 
dv = Av.inflow + max(qv,0) + max(perf(:),0);
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
d = Aa.inflow + Av.inflow + max(qa,0) + max(qv,0);
p = reshape(Grid.volume*Grid.poro,N,1);

% Create matrix vein
AN  = AV + AA;
Ti = AN\p;
Ti = reshape(Ti,Grid.Nx,Grid.Ny,Grid.Nz);

Ta = reshape(V.Ti(1:Grid.N),Grid.Nx,Grid.Ny,Grid.Nz);
Tv = reshape(V.Ti(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny,Grid.Nz);
V.Ta = Ta;  V.Tv = Tv;

function [da,pa,Aa] = maxtof(pa,qa,Aa,da)
% Handling of T -> inf (Boundary region included) :
maxTOF = 50*sum(pa)/sum(qa);
% Find cells that reach max TOF locally
maxIn = max(da);
aboveMax = full(pa./ (max(da, eps*maxIn)) ) > maxTOF;
% Set aboveMax-cells to maxTOF
da(aboveMax)  = maxIn;
pa(aboveMax) = maxIn*maxTOF;
% remove upstream connections
Aa.in(aboveMax(Aa.n(:,2)))  = 0;
Aa.out(aboveMax(Aa.n(:,1))) = 0;
end
function An = Flow(Pa,Grid,Trany,Tranx,Tranz)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; Nz=Grid.Nz; N=Ny*Nx*Nz;
   V.y = zeros(Nx,Ny+1,Nz);
   V.x = zeros(Nx+1,Ny,Nz);
   V.y(:,2:Ny,:) = (Pa(:,1:Ny-1,:)-Pa(:,2:Ny,:)).*Trany(:,2:Ny,:);
   V.x(2:Nx,:,:) = (Pa(1:Nx-1,:,:)-Pa(2:Nx,:,:)).*Tranx(2:Nx,:,:);
   V.z(:,:,2:Nz) = (Pa(:,:,1:Nz-1)-Pa(:,:,2:Nz)).*Tranz(:,:,2:Nz);

   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qz = movmean(V.z,2,3,'Endpoints','discard');
   % define neighbor cell
num = reshape([1:N],Nx,Ny,Nz);
Zin = cat(3,zeros(Nx,Ny,1),num);
Zout = cat(3,num,zeros(Nx,Ny,1));
Yin = [zeros(Nx,1,Nz) num];
Yout = [num zeros(Nx,1,Nz)];
Xin = [zeros(1,Ny,Nz); num];
Xout = [num; zeros(1,Ny,Nz)];
face.neighbors = [reshape(Xin,[],1), reshape(Xout,[],1);reshape(Yin,[],1), reshape(Yout,[],1);reshape(Zin,[],1), reshape(Zout,[],1)];
face.flux = [reshape(V.x,[],1);reshape(V.y,[],1);reshape(V.z,[],1)];
ind = ~any(face.neighbors==0,2); %face neighbors is flatten data
n  = double(face.neighbors(ind,:));
out = min(face.flux(ind), 0);
in  = max(face.flux(ind), 0);

% Cell wise total inflow
inflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);
An.n = n; An.in = in; An.out = out; An.inflow = inflow;
end
end
