function analyze(prm,P,V,Grid,GA,GV,node,vnode,name)
Pan = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);

filename{1} = ['Fig/PA',name];
filename{2} = ['Fig/PV',name];
filename{3} = ['Fig/ToFA',name];
filename{4} = ['Fig/ToFV',name];

figure(1)
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,P,'AlphaData',Grid.B); axis ij; title('Pressure in the capillary bed'); daspect([1 1 1])
caxis([min(P(Grid.B)) max(P(Grid.B))]); set(gca,'color','w');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'Vol [ml/100ml]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'c','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
set(gcf, 'Color', 'w');
export_fig('pawal','-r300'); 
 
figure(2)
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Ta,'AlphaData',Grid.B); title('Time of flight in the capillary bed');daspect([1 1 1])
caxis([min(V.Ta(Grid.B)) max(V.Ta(Grid.B))]); set(gca,'color','w');
hold on
hcb=colorbar
title(hcb,'T [s]')
xlabel('x [mm]')
ylabel('y [mm]')
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
plot(GV,'c','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{3},'-r300'); 
end