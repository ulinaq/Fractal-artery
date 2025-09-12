function analyze(prm,Pp,V,Grid,GA,GV,node,vnode,name)
Pp = Pp*7.7256;
Pan = reshape(Pp(1:Grid.N),Grid.Nx,Grid.Ny);
Pvn = reshape(Pp(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);

filename{1} = ['Fig/Pd',name];
filename{2} = ['Fig/PV',name];
filename{3} = ['Fig/ToFA',name];
filename{4} = ['Fig/ToFV',name];


figure()
% clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pan,'AlphaData',Grid.B); axis ij; title('Pressure in artery compartment'); daspect([1 1 1]);
caxis([min(Pvn(Grid.B)) max(Pan(Grid.B))]); set(gca,'color','w');
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
% set(gcf, 'Color', 'w'); 
% % export_fig(filename{1},'-r300'); 
% 
% 
figure()
% clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pvn,'AlphaData',Grid.B);  title('Pressure in vein compartment');daspect([1 1 1]);
caxis([min(Pvn(Grid.B)) max(Pan(Grid.B))]); %set(gca,'color',0*[1 1 1]);
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
% set(gcf, 'Color', 'w'); 
% % export_fig(filename{2},'-r300'); 

figure()
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Ta,'AlphaData',Grid.B); axis ij; title(['ToF in artery compartment [',name,']']); daspect([1 1 1]);
caxis([0 400]); set(gca,'color','w');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'ToF [s]');
xlabel('x [mm]')
ylabel('y [mm]')
hold on
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
plot(GA,'y','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{1},'-r300'); 


figure()
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Tv,'AlphaData',Grid.B);  title(['ToF in vein compartment[',name,']']);daspect([1 1 1]);
caxis([0 400]); %set(gca,'color',0*[1 1 1]);
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'ToF [s]');
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'g','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{2},'-r300'); 

figure()
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Ta,'AlphaData',Grid.B); axis ij; title(['ToF in artery compartment [',name,']']); daspect([1 1 1]);
caxis([0 400]); set(gca,'color','w');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'ToF [s]');
xlabel('x [mm]')
ylabel('y [mm]')
hold on
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
plot(GA,'y','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{1},'-r300'); 


figure()
clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Tv,'AlphaData',Grid.B);  title(['ToF in vein compartment[',name,']']);daspect([1 1 1]);
caxis([0 400]); %set(gca,'color',0*[1 1 1]);
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'ToF [s]');
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'g','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% daspect([1 1 1])
set(gcf, 'Color', 'w'); 
% export_fig(filename{2},'-r300'); 

end