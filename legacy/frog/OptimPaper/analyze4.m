prm = Frog_parameter();
% prm.Pin = 0.95*prm.Pin;
prm.type = 1;
[Trans,Grid] = FO_Darcy3(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
% segment.rad(74)=segment.rad(74)/2;   % segment that is damaged has much smaller radii
% segment.rad(85)=segment.rad(85)/2;   % segment that is damaged has much smaller radii
% vsegment.rad(223)=vsegment.rad(223)/2;
% vsegment.rad(173)=vsegment.rad(173)/2;
% [Xr,Grid] = FO_Capillary2(node,vnode,Grid);
% Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
% x = xSave(:,1,10);
% x =MObase.x;
x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,2,[]);
% [x,fval] = Frog_nonlinEqOld(node,prm,segment,vnode,vsegment,Xr,1);
% x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
% P = P-Pn;
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
% V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,x);
name = 'A1';
MOA24 = Itungan(x,prm,Grid,node,segment,vnode,Pa,Pv)
analyzex(prm,P,V,Grid,GA,GV,node,vnode,name);
yr = PlotPressure(prm,segment,vsegment,node,x,name);
function [Var] = Itungan(x,prm,Grid,node,segment,vnode,Pa,Pv)
Var.x=x;
Co = node.n + vnode.n; Cv = Co+segment.n;
Var.Q = sum(x([Co+segment.num(segment.isroot)]));
Var.RTot = (prm.Pin-prm.Pout)/Var.Q;
dPa = mean(x(node.isterminal));
dPv = mean(x(node.n+vnode.num(vnode.isterminal)));
dPC = 0.5*(mean(Pv(Grid.B))+ mean(Pa(Grid.B)));
% dPC = 0.5*(max(Pv(Grid.B))+min(Pa(Grid.B)));
Var.RAtree = (prm.Pin-dPa)/(prm.Pin-prm.Pout)*100;
Var.RAcap = std(Pa(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*100;
Var.RVcap = std(Pv(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*100;
Var.RVtree = (dPv-prm.Pout)/(prm.Pin-prm.Pout)*100;
end