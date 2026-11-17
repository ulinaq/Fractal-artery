%% Dual darcy permeability
name = ['Original'];
% Load the modelling parameters
prm = Frog_parameter();             
% Load the image domain and create darcy domain. Then compute transmibility
[Trans,Grid] = FO_Darcy3(prm);      
% Load the segmented vascular graph data
[OriginalNode,OriginalSegment,OriginalVnode,OriginalVsegment] = segOne(prm,Grid,0,0); 
% [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
% Solve Darcy and coupling
[Xr,Grid] = FO_Capillary2(OriginalNode,OriginalVnode,Grid); 
Xr.Pext = Xr.C([OriginalSegment.ext; OriginalVsegment.ext],:);
% Solve the linear system
x = Frog_LinEq(prm,OriginalNode,OriginalSegment,OriginalVnode,OriginalVsegment,Xr.P);
% Solve the nonlinear system
% x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,x1);
% Calculate pressure distribution in Darcy domain
Co = OriginalNode.n + OriginalVnode.n;  Cv = Co + OriginalSegment.n;
in = x([Co+OriginalSegment.num(OriginalSegment.isterminal)]);
out = x([Cv+OriginalVsegment.num(OriginalVsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
% Visualization
GV=graph(OriginalVsegment.init,OriginalVsegment.end,abs(x(Cv+OriginalVsegment.num)));
GA=graph(OriginalSegment.init,OriginalSegment.end,abs(x(Co+OriginalSegment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
V = Flux(prm,Pa,Pv,Grid,Trans,q,OriginalNode,OriginalVnode,OriginalSegment,x);  % Fluxes in Darcy
% yr = PlotPressure(prm,segment,vsegment,node,x1,name);   % Pressure distribution in the whole domain
analyze(prm,P,V,Grid,GA,GV,OriginalNode,OriginalVnode,name);            % Pressure distribution in Darcy domain
%% Numerical experiments
%% Non-linear effect
% Num_experiments();      
%% Occlusions
% NonLinOccl = Occlusions(1); % Nonlinear
% LinOccl = Occlusions(2);    % Linear
% 
% Q = sum(x([Co+segment.num(segment.isroot)])); 
% ReT = (prm.Pin-prm.Pout)/Q;
%     dPa = max(x(node.isterminal));
%     dPv = min(x(node.n+vnode.num(vnode.isterminal)));
%     dPC = 0.5*(mean(Pv(Grid.B))+ mean(Pa(Grid.B)));
% RAt = (prm.Pin-dPa)/(prm.Pin-prm.Pout)*100;
% RAc = std(Pa(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*100;
% RVc = std(Pv(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*100;
% RVt = (dPv-prm.Pout)/(prm.Pin-prm.Pout)*100;
% (prm.Pin-prm.Pout)/Q

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