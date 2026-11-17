% %% Domain Parameters (preparation)
% % global node segment vnode vsegment Grid Trans prm;
prm = Frog_parameter();
[Trans,Grid] = FO_Darcy3(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
% %% Solving flow with damaged vessel
% segment.rad(74)=segment.rad(74)/50;   % segment that is damaged has much smaller radii
% segment.rad(85)=segment.rad(85)/50;   % segment that is damaged has much smaller radii
% vsegment.rad(223)=vsegment.rad(223)/100;
% vsegment.rad(173)=vsegment.rad(173)/100;
% [Xr,Grid,node,vnode] = FO_Capillary2(node,vnode,Grid,prm,segment,vsegment);
% [Xr,Grid,node,vnode] = FO_Capillary(node,vnode,Grid);
[Xr,Grid] = FO_Capillary2(node,vnode,Grid);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% x = Xr.pvec(2*Grid.N+1:end);
Co = node.n + vnode.n;
Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
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
Tin = ones(size(Ti))*max(Ti(Grid.B));
Tin(B) = Ti(B);


save initial.mat node segment vnode vsegment Grid Trans;
%% Optimization
n=2;
lb = -0.3*ones(n,1);
ub = 0.3*ones(n,1);
tic
optPSwpar = optimoptions('particleswarm', 'UseParallel', true, 'UseVectorized', false);
[x1, val] = particleswarm(@FOptim,n,lb,ub,optPSwpar);
toc
x1 = [0.1 -0.05 0.05 0.04];
tta = [0.1 -0.04 0.02];
%Update structure by x perturbation
[nnode,nsegment]=FO_update(x1,tta,node,segment);
[nvnode,nvsegment]=FO_update(x1,tta,vnode,vsegment);

%% Tree visualization
GVn=graph(nvsegment.init,nvsegment.end);
GAn=graph(nsegment.init,nsegment.end);

figure(1)
% imshow(Grid.K)
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pa); title('Time of flight');
daspect([1 1 1])
% hcb=colorbar
% title(hcb,'T [s]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GA,'r','YData',node.XData/prm.pixel,'XData',node.YData/prm.pixel);
daspect([1 1 1])
plot(GAn,'k','YData',nnode.XData*prm.pixel,'XData',nnode.YData*prm.pixel);
daspect([1 1 1])

figure(2)
% imshow(Grid.K)
hold on;
plot(GV,'b','YData',vnode.XData*prm.pixel,'XData',vnode.YData*prm.pixel);
daspect([1 1 1])
plot(GVn,'k','YData',nvnode.XData*prm.pixel,'XData',nvnode.YData*prm.pixel);
daspect([1 1 1])
% title('Artery-Vein Tree');
daspect([1 1 1])