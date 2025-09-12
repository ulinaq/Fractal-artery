%% Simulate dual compartment
clear all; close all;
prm = Frog_parameter(1);
[Trans,Grid] = FD_Darcy(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
% NodeRefining();
[Xr,Grid] = FD_Capillary(node,vnode,segment,vsegment,Grid);
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% [node,segment,x] = FrognonlinEq(node,prm,segment,vnode,vsegment,Xr);
Co = node.n + vnode.n;  Cv = Co + segment.n;    Cq = Cv+vsegment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
% nm = x(Cq+1:end);
q = [in/sum(in)*sum(out);-1*out]; 
P = Xr.C*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
GV = graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
GA = graph(segment.init,segment.end,abs(x(Co+segment.num)));
GV.Edges.width = 10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width = 10*sqrt(abs(GA.Edges.Weight)/sum(in));
% V = Flux2(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,x);
analyze(prm,P,Grid,GA,GV,node,vnode,'trial');
% TracerDual();
% TracerSegment();
% Untitled();