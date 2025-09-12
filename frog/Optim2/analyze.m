x(Cv+vsegment.num(vsegment.isroot))
x(Co+segment.num(segment.isroot))
vsegment.rad(vsegment.isroot)
nnz(x(Cv+vsegment.num(vsegment.isterminal))<0)
min(x(Co+segment.num(segment.isterminal)))

A = imread('frog_domain_low.png');
B = im2bw(A);

aw.dp=max(Pmat(B))-min(Pmat(B));
aw.a1=min(x(node.num(node.isterminal)));
aw.a2=max(x(node.num(node.isterminal)));
aw.v1=min(x(node.n+vnode.num(vnode.isterminal)));
aw.v2=max(x(node.n+vnode.num(vnode.isterminal)));
vw.dp=max(Pmat(B))-min(Pmat(B));
vw.a1=min(x(node.num(node.isterminal)));
vw.a2=max(x(node.num(node.isterminal)));
vw.v1=min(x(node.n+vnode.num(vnode.isterminal)));
vw.v2=max(x(node.n+vnode.num(vnode.isterminal)));

figure()
imagesc(Pmat); colormap(gray); title('Pressure');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar
title(hcb,'[kPa]')
xlabel('x [\times 3.2\cdot10^{-2} mm]')
ylabel('y [\times 3.2\cdot10^{-2} mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
[~,b]=max(Pmat(:));
[I,J] =ind2sub(size(Pmat),b)
sz=60;
scatter(J,I,sz,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor','red',...
              'LineWidth',1,'DisplayName','Maximum pressure')
[~,b]=min(Pmat(:));
[I,J] =ind2sub(size(Pmat),b)
scatter(J,I,sz,'MarkerEdgeColor',[.1 .2 .5],...
              'MarkerFaceColor','cyan',...
              'LineWidth',1,'DisplayName','Minimum pressure')
legend

PP=Pm-Pmat;
figure()
imagesc(PP); colormap(gray); title('Pressure difference');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar
title(hcb,'[kPa]')
xlabel('x [\times 3.2\cdot10^{-2} mm]')
ylabel('y [\times 3.2\cdot10^{-2} mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
[~,b]=max(PP(:));
[I,J] =ind2sub(size(PP),b)
sz=60;
scatter(J,I,sz,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor','red',...
              'LineWidth',1,'DisplayName','Maximum pressure')
[~,b]=min(PP(:));
[I,J] =ind2sub(size(PP),b)
scatter(J,I,sz,'MarkerEdgeColor',[.1 .2 .5],...
              'MarkerFaceColor','cyan',...
              'LineWidth',1,'DisplayName','Minimum pressure')
legend
