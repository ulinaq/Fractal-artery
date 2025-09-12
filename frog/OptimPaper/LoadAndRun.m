name = 'Initial';
load('initial.mat');
[Xr,Grid] = FO_Capillary(node,vnode,Grid); 
% Solve the linear system
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% Solve the nonlinear system
% x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,x1);
% Calculate pressure distribution in Darcy domain
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
% Visualization
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
V = Flux0(Pa,Grid,Trans,q);
% yr = PlotPressure(prm,segment,vsegment,node,x1,name);   % Pressure distribution in the whole domain
analyze(prm,Pa,V,Grid,GA,GV,node,vnode,name);          % Pressure distribution in Darcy domain
