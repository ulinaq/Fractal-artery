I = Vol000000;
Fl = Flo000000;
dVol = (I/100-V.Vol);
figure(1)
clf;
subplot(1,3,1);imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Vol,'AlphaData',Grid.B); caxis([min(dVol(:)) max(I(:))/100]); axis ij; title('Model parameter input'); daspect([1 1 1])
hcb=colorbar;
title(hcb,'Vol [ml/100ml]')
xlabel('x [mm]')
ylabel('y [mm]')
subplot(1,3,2);imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,I/100,'AlphaData',Grid.B); caxis([min(dVol(:)) max(I(:))/100]); axis ij; title('Computed'); daspect([1 1 1])
hcb=colorbar;
title(hcb,'Vol [ml/100ml]')
xlabel('x [mm]')
ylabel('y [mm]')
subplot(1,3,3);imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,dVol,'AlphaData',Grid.B); caxis([min(dVol(:)) max(I(:))/100]); axis ij; title('Difference'); daspect([1 1 1])
hcb=colorbar;
title(hcb,'Vol [ml/100ml]')
xlabel('x [mm]')
ylabel('y [mm]')

dVol = (Fl-V.Flow);
figure(2)
clf;
subplot(1,3,1);imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,V.Flow,'AlphaData',Grid.B); caxis([min(dVol(:)) max(Fl(:))]); axis ij; title('Model parameter input'); daspect([1 1 1])
hcb=colorbar;
title(hcb,'Flow [ml/100ml/min]')
xlabel('x [mm]')
ylabel('y [mm]')
subplot(1,3,2);imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Fl,'AlphaData',Grid.B); caxis([min(dVol(:)) max(Fl(:))]); axis ij; title('Computed'); daspect([1 1 1])
hcb=colorbar;
title(hcb,'Flow [ml/100ml/min]')
xlabel('x [mm]')
ylabel('y [mm]')
subplot(1,3,3);imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,dVol,'AlphaData',Grid.B); caxis([min(dVol(:)) max(Fl(:))]); axis ij; title('Difference'); daspect([1 1 1])
hcb=colorbar;
title(hcb,'Flow [ml/100ml/min]')
xlabel('x [mm]')
ylabel('y [mm]')
 