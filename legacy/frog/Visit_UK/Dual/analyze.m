function analyze(prm,P,Grid,GA,GV,node,vnode,name)
Pan = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pvn = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
filename{1} = ['Fig/PA',name];
filename{2} = ['Fig/PV',name];
filename{3} = ['Fig/ToFA',name];
filename{4} = ['Fig/ToFV',name];

figure(1)
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pan,'AlphaData',Grid.B); axis ij; title('Pressure in the arterial compartment'); daspect([1 1 1])
caxis([min(Pvn(Grid.B)) max(Pan(Grid.B))]); set(gca,'color','w');
hcb=colorbar;
title(hcb,'P [kPa]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'c','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
set(gcf, 'Color', 'w');
% quiver(V.qy,V.qx,'r');
% export_fig(filename{1},'-r300'); 

figure(2)
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pvn,'AlphaData',Grid.B); title('Pressure in the vein compartment');daspect([1 1 1])
caxis([min(Pvn(Grid.B)) max(Pan(Grid.B))]); set(gca,'color','w');
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