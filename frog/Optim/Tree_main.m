% clear all;
% function V=Tree_main(alpha)
tic;
alpha = [0.7,1.4,3,1.2,pi/2,pi/2];
% alpha = xaToFstd;

%% Parameters
prm = Tree_parameter(alpha);
% Grid = Tree_darcyDomain(prm);
% prm.phi(1:2)=alpha(1)/2;
% prm.vphi(1:2)=alpha(2)/2;
% % 
%% Create Fractal tree
% Arterial tree
[node,segment] = Tree_artery(prm);
% venous tree
[vnode,vsegment] = Tree_vein(prm);
% Tree visualization
GV=graph(vsegment.init,vsegment.end);
GA=graph(segment.init,segment.end);

figure()
plot(GV,'XData',vnode.XData,'YData',vnode.YData);
daspect([1 1 1])
% axis([0 140 0 160])
hold on;
plot(GA,'XData',node.XData,'YData',node.YData);
title('Artery-Vein Tree');
daspect([1 1 1])

%% Capillary Domain
% domain size 
[Xr,T,Grid,node,vnode] = Tree_Capillary_Darcy2(node, vnode);

%% Calculate pressure in nodes (Update April 26, 2018)
[node,segment,x] = Tree_nonlinEq(node,prm,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:Co;
q = [x([Co+seg(node.in)-1]);-1*x([Cv+seg(vnode.out)-1])];
Pmat = x(Co)+Xr.pvec*q;
Pmat = reshape(Pmat,Grid.Nx,Grid.Ny);
Xr.pressure = Pmat;
V = Flux(Pmat,Grid,T,q);

Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);

figure()
contourf(Ti');  colorbar; title('Time of Flight');
daspect([1 1 1])

figure()
contourf(Pmat'); colorbar; title('Pressure');
hold on;
quiver(V.qx',V.qy','r'); %title('Fluxes');
toc

% clear Co Cv q seg Pmat

%% Calculate Flux
function V = Flux(Pmat,Grid,Trans,q)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
       
   V.y = zeros(Nx,Ny+1);
   V.x = zeros(Nx+1,Ny);;
   V.y(:,2:Ny) = (Pmat(:,1:Ny-1)-Pmat(:,2:Ny)).*Trans.y(:,2:Ny);
   V.x(2:Nx,:) = (Pmat(1:Nx-1,:)-Pmat(2:Nx,:)).*Trans.x(2:Nx,:);

   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.u = sqrt(V.qx.^2+V.qy.^2);
   u = reshape(V.u,N,1);
   V.range = max(u)-min(u);
   
   qq=[];
for i=1:length(q)
    qq=[qq; Grid.src.rate{i}*q(i)];
end

qi = sparse(Grid.src.cell,1, qq, N,1);
pv = reshape(Grid.volume.*Grid.poro,N,1);
% define neighbor
num = reshape([1:N],Nx,Ny);
Yin = [zeros(Nx,1) num];
Yout = [num zeros(Nx,1)];
Xin = [zeros(1,Ny); num];
Xout = [num; zeros(1,Ny)];
face.neighbors = [reshape(Xin,[],1), reshape(Xout,[],1);reshape(Yin,[],1), reshape(Yout,[],1)];
face.flux = [reshape(V.x,[],1);reshape(V.y,[],1)];
ind = ~any(face.neighbors==0,2);
n  = double(face.neighbors(ind,:));

out = min(face.flux(ind), 0);
in  = max(face.flux(ind), 0);

% Cell wise total inflow
inflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);

% diagonal 
d = inflow + max(qi,0);

% Inflow flux matrix
A  = sparse(n(:,2), n(:,1),  in, N, N)...
   + sparse(n(:,1), n(:,2), -out, N, N);
A = -A + spdiags(d, 0, N, N);

% Solve
V.Ti = A\pv;
  
end

% end