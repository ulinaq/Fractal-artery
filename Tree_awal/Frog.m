%% Domain Parameters (preparation) 3D domain
% global node segment vnode vsegment Grid Trans prm;
prm = Frog_parameter();
[node,vnode,segment,vsegment] = Tree(prm);
[Trans,Grid,Xr] = FO_Darcy(prm,node,vnode);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
% Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny,Grid.Nz);
% Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny,Grid.Nz);
% GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
% GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
% GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
% GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
% V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode);
analyze(prm,P,Grid);


%% Simulate dual permeability
prm = Frog_parameter();
[Trans,Grid] = FO_Darcy3(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = seggg(prm,Grid,0,0);
% segment.rad(74)=segment.rad(74)/50;   % segment that is damaged has much smaller radii
% segment.rad(85)=segment.rad(85)/50;   % segment that is damaged has much smaller radii
% vsegment.rad(223)=vsegment.rad(223)/100;
% vsegment.rad(173)=vsegment.rad(173)/100;
[Xr,Grid] = FO_Capillary2(node,vnode,Grid);
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode);
analyze(prm,P,V,Grid,GA,GV,node,vnode,'trial');
as=[min(Pa(Grid.B)) max(Pa(Grid.B));min(Pv(Grid.B)) max(Pv(Grid.B));min(V.Ta(Grid.B)) max(V.Ta(Grid.B));min(V.Tv(Grid.B)) max(V.Tv(Grid.B))];
% Change radius based on pressure
% [xN,Pnew,rad]=FO_Rad(x,node,segment,vnode,vsegment,Pa,Pv,Grid,Xr,prm,Trans);

%% Elasticity
in = xC([Co+segment.num(segment.isterminal)]);
out = xC([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+xC(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode);

prm = Frog_parameter();
[Trans,Grid] = FO_Darcy3(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = seggg(prm,Grid,0,0);
% NA = segment.num(segment.isroot);
% segment.rad(NA(2))=segment.rad(NA(2))*5/100;   % segment that is damaged has much smaller radii
% segment.rad(74)=segment.rad(74)/50;   % segment that is damaged has much smaller radii
% vsegment.rad(223)=vsegment.rad(223)/100;
% segment.rad(85)=segment.rad(85)/50;   % segment that is damaged has much smaller radii
% vsegment.rad(173)=vsegment.rad(173)/100;
% segment.rad(46)=segment.rad(46)/50;   % segment that is damaged has much smaller radii
% segment.rad(20)=segment.rad(85)/50;   % segment that is damaged has much smaller radii
% vsegment.rad(223)=vsegment.rad(223)/100;
% vsegment.rad(173)=vsegment.rad(173)/100;
[Xr,Grid] = FO_Capillary2(node,vnode,Grid);
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
[node,segment,x1] = Frog_nonlinEq2(node,prm,segment,vnode,vsegment,Xr);
in = x1([Co+segment.num(segment.isterminal)]);
out = x1([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x1(end);
Pan = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pvn = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
GV=graph(vsegment.init,vsegment.end,abs(x1(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(x1(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
Vn = Flux(prm,Pan,Pvn,Grid,Trans,q,node,vnode);
analyze(prm,P,Vn,Grid,GA,GV,node,vnode,['_EJ']);
analyze2();

% Compute Time of Flight
V = Flux0(Pa,Grid,Trans,q);
Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);
% Tin = ones(size(Ti))*max(Ti(Grid.B));
% Tin(B) = Ti(B);
figure()
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Ti); title('Time of flight');
daspect([1 1 1])
hcb=colorbar;
title(hcb,'T [s]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])

save initial.mat node segment vnode vsegment Grid Trans;
%% Optimization
n=2;
lb = -0.3*ones(n,1);
ub = 0.3*ones(n,1);
tic
optPSwpar = optimoptions('particleswarm', 'UseParallel', true, 'UseVectorized', false);
[x1, val] = particleswarm(@FOptim,n,lb,ub,optPSwpar);
toc

%Update structure by x perturbation
[node,segment]=FO_update(x1,node,segment);
[vnode,vsegment]=FO_update(x1,vnode,vsegment);

%% Tree visualization
GVn=graph(vsegment.init,vsegment.end);
GAn=graph(segment.init,segment.end);

figure()
imshow(Grid.K)
hold on;
plot(GVn,'YData',vnode.XData,'XData',vnode.YData);
daspect([1 1 1])
plot(GAn,'YData',node.XData,'XData',node.YData);
% title('Artery-Vein Tree');
daspect([1 1 1])