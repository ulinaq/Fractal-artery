%% Domain Parameters (preparation)
[T,Grid] = Frog_Darcy();    % Load the image domain and create darcy domain. Then compute transmibility
A = imread('frog_domain_low.png');
B = im2bw(A);
prm = Frog_parameter()
load('frog_trees.mat');
[node,segment] = seggg(tree_a{1},1);
[vnode,vsegment] = seggg(tree_v{1},2);
%% tambahan (nanti) ra>0.65, rv>0.95
% vsegment.init([end+1,end+2])=[7,9];
% vsegment.end([end+1,end+2])=[6,8];
% vsegment.len([end+1,end+2])=vsegment.len(vsegment.end==7);
% vsegment.rad([end+1,end+2])=vsegment.rad(vsegment.end==7);
% vsegment.n=length(vsegment.len);
% vnode.junction(7)=1;
% vnode.isterminal([9,7])=0;
%% ra>1
segment.init(end+1)=10;
node.isterminal(10)=0;
segment.end(end+1)=4;
segment.len(end+1)=segment.len(segment.end==10);
segment.rad(end+1)=segment.rad(segment.end==10);
segment.n=length(segment.len);

save initial.mat node segment vnode vsegment B Grid Trans Amat;
%% Tree visualization
GV=graph(vsegment.init,vsegment.end,vsegment.len);
GA=graph(segment.init,segment.end,segment.len);
%% note segment root and terminal
segment = segroot(node,segment);
vsegment = segroot(vnode,vsegment);

figure()
imshow(B)
hold on;
plot(GV,'YData',vnode.XData,'XData',vnode.YData);
daspect([1 1 1])
plot(GA,'YData',node.XData,'XData',node.YData);
title('Artery-Vein Tree');
daspect([1 1 1])

[Xr,Trans,Grid,node,vnode] = Frog_Capillary(node, vnode);

x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% [node,segment,x] = Frog_nonlinEq(node,prm,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:Co;
in = x([Co+seg(segment.isterminal)]);
out = x([Cv+seg(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
Pmat = x(Co)+Xr.pvec*q;
Pmat = reshape(Pmat,Grid.Nx,Grid.Ny);
Xr.pressure = Pmat;

A = imread('frog_domain_low.png');
B = im2bw(A);
Bz = size(A);
aa = 1-B;
aa = reshape(aa,prod(Bz),1);

P = ones(size(B))*min(min(Pmat));
P(B) = Pmat(B);
V = Flux(Pmat,Grid,Trans,q,aa);
Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);
Tin = ones(size(Ti))*min(V.Ti(V.Ti>1));
Tin(B) = Ti(B);
figure()
contourf(log(Tin)); colormap(gray); colorbar; title('Time of Flight');
daspect([1 1 1])

figure()
% imagesc(P); colorbar; title('Pressure');
contourf(P); colormap(gray); colorbar; title('Pressure');
hold on
plot(GV,'YData',vnode.XData,'XData',vnode.YData);
daspect([1 1 1])
plot(GA,'YData',node.XData,'XData',node.YData);
title('Artery-Vein Tree');
daspect([1 1 1])

figure()
contourf(P); colormap(gray); colorbar; title('Pressure');
daspect([1 1 1])
hold on;
plot(GV,'YData',vnode.XData,'XData',vnode.YData);
daspect([1 1 1])
plot(GA,'YData',node.XData,'XData',node.YData);
title('Artery-Vein Tree');
daspect([1 1 1])
quiver(V.qy,V.qx,'g'); %title('Fluxes');

