function analyzex(prm,Pp,Grid,GA,GV,node,vnode,name)
Pp = Pp*7.5006;
Pan = reshape(Pp(1:Grid.N),Grid.Nx,Grid.Ny);
Pvn = reshape(Pp(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);

filename{1} = ['Fig/PAd',name];
filename{2} = ['Fig/PVd',name];

figure('Name',name)
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pan,'AlphaData',Grid.B); axis ij; title('Pressure drop in artery compartment'); daspect([1 1 1]);
caxis([min(Pan(Grid.B)) max(Pan(Grid.B))]); set(gca,'color','w');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'P [mmHg]');
xlabel('x [mm]')
ylabel('y [mm]')
hold on
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{1},'-r300'); 


figure('Name',name)
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pvn,'AlphaData',Grid.B);  title('Pressure drop in vein compartment');daspect([1 1 1]);
caxis([min(Pvn(Grid.B)) max(Pvn(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'P [mmHg]');
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{2},'-r300'); 
end