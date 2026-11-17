function analyze(prm,P,Grid)
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny,Grid.Nz);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny,Grid.Nz);
figure(1);clf;
[y,z] = meshgrid(linspace(0,50*prm.pixel,50));
for off=5:10:Grid.Nz   
  x = off + zeros(size(z));
    % My standin for your vorticity data
    c = Pa(:,:,off);
    surf(x,y,z,c,'FaceAlpha',0.9,'EdgeColor','none')
    caxis([min(P) max(P)]); set(gca,'color','w');
    hold on
end
hcb=colorbar;
title(hcb,'P [kPa]')
xlabel('z [mm]')
ylabel('y [mm]')
zlabel('x [mm]')
hold off
  xlim([0 Grid.Nz])
  set(gcf, 'Color', 'w'); 
  export_fig('Fig/Pa','-r300'); 
figure(2);clf;
[y,z] = meshgrid(linspace(0,50*prm.pixel,50));
for off=5:10:Grid.Nz    
  x = off + zeros(size(z));
    % My standin for your vorticity data
    c = Pv(:,:,off);
    surf(x,y,z,c,'FaceAlpha',0.9,'EdgeColor','none')
    caxis([min(P) max(P)]); set(gca,'color','w');
    hold on
end
hcb=colorbar;
title(hcb,'P [kPa]')
xlabel('z [mm]')
ylabel('y [mm]')
zlabel('x [mm]')

  hold off
  xlim([0 Grid.Nz])
  set(gcf, 'Color', 'w'); 
  export_fig('Fig/Pv','-r300'); 

end