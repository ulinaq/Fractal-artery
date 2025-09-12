%% Domain Parameters
[T,Grid] = Optim_Darcy();

%% Start optimasi ---->>> Optim function
% Variable for optimization
% (1) gamma art, (2) gamma vein, (3) L/R, (4) vein/art, (5) initial theta artery,
% (6) initial theta vein, (9) bif. exp.
% lb = [0.3,0.3,5,1,pi/2,-pi/2];
% ub = [10/3,10/3,15,3,pi,0];
% 20/07/2018 : add alpha(7) and (8) for root terminal variables
lb = [0.3,0.3,5,1,pi/7,9/8*pi,-70,-70,2,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3];
ub = [10/3,10/3,15,3,6/7*pi,15/8*pi,70,70,3,10/3,10/3,10/3,10/3,10/3,10/3,10/3,10/3,10/3,10/3,10/3];

tic
optPSwpar = optimoptions('particleswarm', 'UseParallel', true, 'UseVectorized', false);
[xaTSw, valToF] = particleswarm(@OptimTreeDomain,20,lb,ub);
% optPSepar = optimoptions('patternsearch','UseParallel', true, 'UseCompletePoll', true, 'UseVectorized', false, 'MaxFunctionEvaluations', 50000,'MaxIterations', 20000, 'MeshTolerance',1e-7);
% [xptsc2,fval]=patternsearch(@OptimTreeDomain,xaT,[],[],[],[],lb,ub,optPSepar);
toc
prm = Tree_parameter(xptsc2);
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
axis([-70 70 0 160])
xlabel('x')
ylabel('y')
hold on;
plot(GA,'XData',node.XData,'YData',node.YData);
title('Artery-Vein Tree');
daspect([1 1 1])
%% Solve darcy equation in Capillary Domain
[Xr,Grid,node,vnode] = Optim_Capillary(node, vnode);

%% Calculate pressure in nodes (Update April 26, 2018)
[node,segment,x] = Tree_nonlinEq(node,prm,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:Co;
q = [x([Co+seg(node.in)-1]);-1*x([Cv+seg(vnode.out)-1])];
Pmat = x(Co)+Xr.pvec*q;
Pmat = reshape(Pmat,Grid.Size(1),Grid.Size(2));
Xr.pressure = Pmat;
V = Flux(Pmat,Grid,T,q);

Ti = reshape(V.Ti,Grid.Size(1),Grid.Size(2));

figure()
contourf(Grid.K');  colorbar; title('Permeability');
daspect([1 1 1])
xlabel('x')
ylabel('y')

figure()
contourf(log10(Ti'));  colorbar; title('Time of Flight');
daspect([1 1 1])
xlabel('x')
ylabel('y')

figure()
contourf(Pmat',20); colorbar; title('Pressure');
hold on;
quiver(V.qx',V.qy','r'); %title('Fluxes');
xlabel('x')
ylabel('y')
%% Calculate Flux
function V = Flux(Pmat,Grid,Trans,q)
   % Fluxes at staggered grid
   Nx=Grid.Size(1); Ny=Grid.Size(2); N=Ny*Nx;
       
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
