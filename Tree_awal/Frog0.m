%% Domain Parameters (preparation)
% global node segment vnode vsegment Grid Trans prm;
prm = Frog_parameter();
[Trans,Grid] = FO_Darcy(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = seggg(0,0);
% %% Solving flow with damaged vessel
% segment.rad(74)=segment.rad(74)/50;   % segment that is damaged has much smaller radii
% segment.rad(85)=segment.rad(85)/50;   % segment that is damaged has much smaller radii
vsegment.rad(223)=vsegment.rad(223)/100;
% vsegment.rad(173)=vsegment.rad(173)/100;
% [Xr,Grid,node,vnode] = FO_Capillary2(node,vnode,Grid,prm,segment,vsegment);
[Xr,Grid,node,vnode] = FO_Capillary(node,vnode,Grid);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% x = Xr.pvec(2*Grid.N+1:end);
Co = node.n + vnode.n;
Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
% Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
% add flow rate as weight on graph plot
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+1:Cv+vsegment.n)));
GA=graph(segment.init,segment.end,abs(x(Co+1:Cv)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
% Change radius based on pressure
% [xU,Pnew,rad]=FO_Rad(x,node,segment,vnode,vsegment,Pmat,Grid,Xr,prm,Trans);
% Compute Time of Flight
V = Flux0(Pa,Grid,Trans,q);
Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);
% Tin = ones(size(Ti))*max(Ti(Grid.B));
% Tin(B) = Ti(B);
figure()
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Ti); title('Time of flight');
daspect([1 1 1])
hcb=colorbar
title(hcb,'T [s]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'b','YData',vnode.XData*prm.pixel,'XData',vnode.YData*prm.pixel,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData*prm.pixel,'XData',node.YData*prm.pixel,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
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