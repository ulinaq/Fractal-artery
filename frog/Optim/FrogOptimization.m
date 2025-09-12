  %% Dual darcy permeability
name = ['Trial'];
% Load the modelling parameters
prm = Frog_parameter();             
% Load the image domain and create darcy domain. Then compute transmibility
[Trans,Grid] = FO_Darcy3(prm);      
% Load the segmented vascular graph data
[node,segment,vnode,vsegment] = SetVessel2(prm, Grid, 20);
while all(Grid.B([node.input; vnode.output])) == 0
[node,segment,vnode,vsegment] = SetVessel2(prm, Grid, 20);
1
end
% [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
% Solve Darcy and coupling
[Xr,Grid] = FO_Capillary2(node,vnode,Grid); 
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
% Solve the linear system
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% Solve the nonlinear system
% x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,x1);
% Calculate pressure distribution in Darcy domain
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
% Visualization
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,x);  % Fluxes in Darcy
% yr = PlotPressure(prm,segment,vsegment,node,x1,name);   % Pressure distribution in the whole domain
analyze(prm,P,V,Grid,GA,GV,node,vnode,name);            % Pressure distribution in Darcy domain
% save init.mat node segment vnode vsegment Grid Trans;

function [node,segment,vnode,vsegment] = SetVessel2(prm, Grid, n)
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
var = 0.3*(rand(1,n)-0.5);
[node,segment]=FO_update(var(1:round(n/4)),var(round(n/4)+1:round(n/2)),node,segment,Grid);
[vnode,vsegment]=FO_update(var(round(n/2)+1:round(3*n/4)), var(round(3*n/4)+1:end),vnode,vsegment,Grid);
end 

function [node,segment,vnode,vsegment] = SetVessel(prm, Grid)
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
% the initial vascular structure
dxA = 0.2*(rand(1,node.n)-0.5);
dThetaA = 0.2*(rand(1,segment.n)-0.5);
dxV = 0.2*(rand(1,vnode.n)-0.5);
dThetaV = 0.2*(rand(1,vsegment.n)-0.5);
%Update structure by x perturbation
[node,segment]=FO_update(dxA,dThetaA,node,segment,Grid);
[vnode,vsegment]=FO_update(dxV,dThetaV,vnode,vsegment,Grid);
end