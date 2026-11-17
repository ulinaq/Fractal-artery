%% Dual darcy permeability
name = ['Original'];
% Load the modelling parameters
prm = Frog_parameter();             
% Load the image domain and create darcy domain. Then compute transmibility
[Trans,Grid] = FO_Darcy(prm);      
% Load the segmented vascular graph data
[OriginalNode,OriginalSegment,OriginalVnode,OriginalVsegment] = segOne(prm,Grid,0,0); 
% [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
% Solve Darcy and coupling
[Xr,Grid] = FO_Capillary(OriginalNode,OriginalVnode,Grid); 
% Solve the linear system
x = Frog_LinEq(prm,OriginalNode,OriginalSegment,OriginalVnode,OriginalVsegment,Xr.P);
% Calculate pressure distribution in Darcy domain
Co = OriginalNode.n + OriginalVnode.n;  Cv = Co + OriginalSegment.n;
in = x([Co+OriginalSegment.num(OriginalSegment.isterminal)]);
out = x([Cv+OriginalVsegment.num(OriginalVsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
% Visualization
GV=graph(OriginalVsegment.init,OriginalVsegment.end,abs(x(Cv+OriginalVsegment.num)));
GA=graph(OriginalSegment.init,OriginalSegment.end,abs(x(Co+OriginalSegment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
V = Flux0(Pa,Grid,Trans,q);
% V = Flux(prm,Pa,Pv,Grid,Trans,q,OriginalNode,OriginalVnode,OriginalSegment,x);  % Fluxes in Darcy
% yr = PlotPressure(prm,segment,vsegment,node,x1,name);   % Pressure distribution in the whole domain
analyze(prm,Pa,V,Grid,GA,GV,OriginalNode,OriginalVnode,name, Pset, Tset)           % Pressure distribution in Darcy domain


function [node,segment,vnode,vsegment] = SetVessel(prm, Grid)
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
% the initial vascular structure
dxA = 0.1*(rand(1,node.n)-0.5);
dThetaA = 0.35*(rand(1,segment.n)-0.5);
dxV = 0.1*(rand(1,vnode.n)-0.5);
dThetaV = 0.35*(rand(1,vsegment.n)-0.5);
%Update structure by x perturbation
[node,segment]=FO_update(dxA,dThetaA,node,segment,Grid);
[vnode,vsegment]=FO_update(dxV,dThetaV,vnode,vsegment,Grid);
end