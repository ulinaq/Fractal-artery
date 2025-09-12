%% Simulate one compartment
prm = Frog_parameter();
[Trans,Grid] = FS_Darcy(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
NodeRefining();
[Xr,Grid] = FS_Capillary(node,vnode,segment,vsegment,Grid);
% Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out]; 
P = Xr.C*q+x(end);
P = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
% V = Flux(prm,P,Grid,Trans,q,node,vnode,segment,vsegment,x);
% analyze(prm,P,V,Grid,GA,GV,node,vnode,'trial');
% TracerSegment;
% Untitled();
% Coarsening();