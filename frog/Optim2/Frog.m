%% Domain Parameters (preparation)
% global node segment vnode vsegment Grid Trans prm;
prm = Frog_parameter();
[Trans,Grid] = FO_Darcy(prm);    % Load the image domain and create darcy domain. Then compute transmibility
% [node,segment,vnode,vsegment] = Tree_edit(ra,rv);
[node,segment,vnode,vsegment] = seggg(0,0);
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

%% Solving flow with damaged vessel
segment.rad(74)=segment.rad(74)*10;   % segment that is damaged has much smaller radii
vsegment.rad(223)=vsegment.rad(223)/10;

[Xr,Grid,node,vnode] = FO_Capillary(node,vnode,Grid);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% [node,segment,x] = Frog_nonlinEq(node,prm,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
Pmat = x(end)+Xr.pvec*q;
Pmat = reshape(Pmat,Grid.Nx,Grid.Ny);
Xr.pressure = Pmat;
% add flow rate as weight on graph plot
GV=graph(vsegment.init,vsegment.end,x(Cv+1:end-1));
GA=graph(segment.init,segment.end,x(Co+1:Cv));
GV.Edges.width=20*abs(GV.Edges.Weight)/sum(out);
GA.Edges.width=20*abs(GA.Edges.Weight)/sum(in);

figure()
imshow(Grid.K)
hold on;
plot(GV,'YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width);
daspect([1 1 1])
plot(GA,'YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width);
daspect([1 1 1])

A = imread('frog_domain_low.png');
B = im2bw(A);
Bz = size(A);
aa = 1-B;
aa = reshape(aa,prod(Bz),1);

P = ones(size(B))*max(max(Pmat));
P(B) = Pmat(B);
V = Flux(Pmat,Grid,Trans,q,aa);
Ti = reshape(1./V.u,Grid.Nx,Grid.Ny);
Tin = ones(size(Ti))*min(V.Ti(V.Ti>1));
Tin(B) = Ti(B);
figure()
imagesc(Pmat); colormap(gray); title('Pressure');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar
title(hcb,'[kPa]')
xlabel('x [\times 3.2\cdot10^{-2} mm]')
ylabel('y [\times 3.2\cdot10^{-2} mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width);
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width);
daspect([1 1 1])


quiver(V.qy,V.qx,'g'); %title('Fluxes');