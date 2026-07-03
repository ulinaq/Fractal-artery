name = ['OptimalFlux'];
% Load the modelling parameters
prm = Frog_parameter();             
load('init.mat');
%% Optimization
% n=node.n+segment.n+vnode.n+vsegment.n;
n = 20;
lb = -0.1*ones(n,1);
ub = 0.1*ones(n,1);
tic
optPSwpar = optimoptions('particleswarm', 'UseParallel', true, 'MaxIterations', 200);
[var, val] = particleswarm(@FOptim,n,lb,ub,optPSwpar);
toc

%Update structure by x perturbation
[OptimNode,Optimsegment]=FO_update(var(1:round(n/4)),var(round(n/4)+1:round(n/2)),node,segment,Grid);
[OptimVnode,Optimvsegment]=FO_update(var(round(n/2)+1:round(3*n/4)), var(round(3*n/4)+1:end),vnode,vsegment,Grid);

% [OptimNode,Optimsegment]=FO_update(var(1:node.n),var(node.n+1:node.n+segment.n),node,segment,Grid);
% [OptimVnode,Optimvsegment]=FO_update(var(node.n+segment.n+1:node.n+segment.n+vnode.n), var(node.n+segment.n+vnode.n+1:end),vnode,vsegment,Grid);

[Xr,Grid] = FO_Capillary2(OptimNode,OptimVnode,Grid); 
Xr.Pext = Xr.C([Optimsegment.ext; Optimvsegment.ext],:);
% Solve the linear system
x = Frog_LinEq(prm,OptimNode,Optimsegment,OptimVnode,Optimvsegment,Xr.P);
% Calculate pressure distribution in Darcy domain
Co = OptimNode.n + OptimVnode.n;  Cv = Co + Optimsegment.n;
in = x([Co+Optimsegment.num(Optimsegment.isterminal)]);
out = x([Cv+Optimvsegment.num(Optimvsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
% Visualization
GV=graph(Optimvsegment.init,Optimvsegment.end,abs(x(Cv+Optimvsegment.num)));
GA=graph(Optimsegment.init,Optimsegment.end,abs(x(Co+Optimsegment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
V = Flux(prm,Pa,Pv,Grid,Trans,q,OptimNode,OptimVnode,Optimsegment,x);  % Fluxes in Darcy
% yr = PlotPressure(prm,segment,vsegment,node,x1,name);   % Pressure distribution in the whole domain
analyze(prm,P,V,Grid,GA,GV,OptimNode,OptimVnode,name);            % Pressure distribution in Darcy domain
save OptimalFlux.mat var;