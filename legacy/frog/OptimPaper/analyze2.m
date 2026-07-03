tic
% prm = Frog_parameter();
% [Trans,Grid] = FO_Darcy3(prm);    % Load the image domain and create darcy domain. Then compute transmibility
% [node,segment,vnode,vsegment] = seggg(prm,Grid,0,0);
% [Xr,Grid] = FO_Capillary2(node,vnode,Grid);
% Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
% [node,segment,x] = Frog_nonlinEq2(node,prm,segment,vnode,vsegment,Xr);
% % x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% Co = node.n + vnode.n;  Cv = Co + segment.n;
% in = x([Co+segment.num(segment.isterminal)]);
% out = x([Cv+vsegment.num(vsegment.isterminal)]);
% q = [in/sum(in)*sum(out);-1*out];
% P = Xr.C*q+x(end);
% Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
% Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
% V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode);
% severed segment
% NN = 5:5:95;
NA = vsegment.num(vsegment.isroot);
% XRV=x;
% for i=1:length(NN)
[node,segment,vnode,vsegment] = seggg(prm,Grid,0,0);
vsegment.rad(NA(1))=vsegment.rad(NA(1))*5/100;   % segment that is damaged has much smaller radii
[node,segment,xN] = Frog_nonlinEq2(node,prm,segment,vnode,vsegment,Xr);
% [node,segment,xN] = Frog_nonLinEq(node,prm,segment,vnode,vsegment,Xr.P);
in = xN([Co+segment.num(segment.isterminal)]);
out = xN([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+xN(end);
Pan = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pvn = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
Vn = Flux(prm,Pan,Pvn,Grid,Trans,q,node,vnode);
GV=graph(vsegment.init,vsegment.end,abs(xN(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(xN(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
Vn = Flux(prm,Pan,Pvn,Grid,Trans,q,node,vnode);
analyze(prm,P,Vn,Grid,GA,GV,node,vnode,['_R1-5inv']);

% 
% XRV=[XRV,xN];
% name = 'Elast';
name=['VeinR1-5inv'];
Res{i} = Itungan(xN,name,Grid,Pa,Pv,Pan,Pvn,V,Vn,node,prm,segment,vnode,vsegment,GA,GV,Xr);
var{i,1} = ['Model_',name];
var{i,2} = vpa(sym([Res{i}.Q,Res{i}.RT,Res{i}.aM,Res{i}.vM]),4);
var{i,3} = [vpa(sym(Res{i}.PaM),4),vpa(sym(Res{i}.PaS),4)];
var{i,4} = [vpa(sym(Res{i}.PvM),4),vpa(sym(Res{i}.PvS),4)];
var{i,5} = [vpa(sym(Res{i}.TaM),4),vpa(sym(Res{i}.TaS),4)];
var{i,6} = [vpa(sym(Res{i}.TvM),4),vpa(sym(Res{i}.TvS),4)];
toc
% end


% latex_table = latex(vpa(sym(varRA),3));
toc
function [Var] = Itungan(x,name,Grid,Pa,Pv,Pan,Pvn,V,Vn,node,prm,segment,vnode,vsegment,GA,GV,Xr)
Var.name=name;
Var.range= [min(Pan(Grid.B)) max(Pan(Grid.B))];
Var.x=x;
[Var.a,Var.v] = Resistance(node,prm,segment,vnode,vsegment,Xr,x);
Co = node.n + vnode.n; Cv = Co+segment.n;
Var.Q = sum(x([Co+segment.num(segment.isroot)]));
Var.RT = (prm.Pin-prm.Pout)/Var.Q;
Pad = Pan - Pa;
Pvd = Pvn - Pv;
Tad = Vn.Ta - V.Ta;
Tvd = Vn.Ta - V.Tv;
Var.PaM=mean(Pad(:));
Var.PaS=std(Pad(:));
Var.TaM=mean(Tad(:));
Var.TaS=std(Tad(:));
Var.PvM=mean(Pvd(:));
Var.PvS=std(Pvd(:));
Var.TvM=mean(Tvd(:));
Var.TvS=std(Tvd(:));
Var.aM=mean(Var.a);
Var.vM=mean(Var.v);
% 
% clf;
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pad,'AlphaData',Grid.B); axis ij; title('\Delta P in artery compartment'); daspect([1 1 1])
% caxis([min(Pad(Grid.B)) max(Pad(Grid.B))]);% set(gca,'color',0*[1 1 1]);
% % contourf(P); colormap(gray); colorbar; title('Pressure');
% hcb=colorbar;
% title(hcb,'P [kPa]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% hold on
% % plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% % daspect([1 1 1])
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/Pad_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pvd,'AlphaData',Grid.B);  title('\Delta P in vein compartment');daspect([1 1 1])
% caxis([min(Pvd(Grid.B)) max(Pvd(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% % contourf(P); colormap(gray); colorbar; title('Pressure');
% hcb=colorbar;
% title(hcb,'P [kPa]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% hold on
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
% % plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% % daspect([1 1 1])
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/Pvd_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Tad,'AlphaData',Grid.B); title('\Delta ToF in artery compartment');
% daspect([1 1 1]); caxis([min(Tad(Grid.B)) max(Tad(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% hold on
% hcb=colorbar
% title(hcb,'T [s]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% daspect([1 1 1])
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/Tad_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Tvd,'AlphaData',Grid.B); title('\Delta ToF in vein compartment');
% daspect([1 1 1]); caxis([min(Tvd(Grid.B)) max(Tvd(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% hold on
% hcb=colorbar
% title(hcb,'T [s]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/Tvd_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pan,'AlphaData',Grid.B); axis ij; title('Pressure in artery compartment'); daspect([1 1 1])
% caxis([min(Pvn(Grid.B)) max(Pan(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% % contourf(P); colormap(gray); colorbar; title('Pressure');
% hcb=colorbar;
% title(hcb,'P [kPa]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% hold on
% % plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% % daspect([1 1 1])
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/Part_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pvn,'AlphaData',Grid.B);  title('Pressure in vein compartment');daspect([1 1 1])
% caxis([min(Pvn(Grid.B)) max(Pan(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% % contourf(P); colormap(gray); colorbar; title('Pressure');
% hcb=colorbar;
% title(hcb,'P [kPa]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% hold on
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
% % plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% % daspect([1 1 1])
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/Pvein_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Vn.Ta,'AlphaData',Grid.B); title('ToF in artery compartment');
% daspect([1 1 1]); caxis([min(Vn.Ta(Grid.B)) max(Vn.Ta(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% hold on
% hcb=colorbar
% title(hcb,'T [s]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% daspect([1 1 1])
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/ToFa_',name],'-r300'); 
% clf;
% 
% figure(1)
% imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Vn.Tv,'AlphaData',Grid.B); title('ToF in vein compartment');
% daspect([1 1 1]); caxis([min(Vn.Ta(Grid.B)) max(Vn.Ta(Grid.B))]); %set(gca,'color',0*[1 1 1]);
% hold on
% hcb=colorbar
% title(hcb,'T [s]')
% xlabel('x [mm]')
% ylabel('y [mm]')
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
% set(gcf, 'Color', 'w'); 
% export_fig(['Fig/ToFv_',name],'-r300'); 
end

function [Rart,Rvein] = Resistance(node,prm,segment,vnode,vsegment,Xr,x)
Co = node.n + vnode.n; Cv = Co+segment.n;
xend = Cv+vsegment.n+1;
in = node.num(node.isterminal); 
out = vnode.num(vnode.isterminal);
Nin = nnz(in); Nout = nnz(out); k=1;
Rart = zeros(segment.n,1); Rvein = zeros(vsegment.n,1);
% Artery 
for i = 1 : segment.n
    segint = segment.init(i);
    segend = segment.end(i);
    if node.junction(segint)
        par = segment.num(segment.end==segint);
        cphi = cos(3*(segment.Gtheta(i)-segment.Gtheta(par))/4);
        Rart(i) = (8*prm.mu*segment.len(i)*x(Co+i)/(pi*segment.rad(i)^4)...
            *(1-3*(x(segint)/2 + x(segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.ha*prm.E))...
            +  prm.Rho*x(Co+par)^2/(2*pi^2*segment.rad(par)^4)...
            *(1+x(Co+i)^2*segment.rad(par)^4/(x(Co+par)^2*segment.rad(i)^4)...
            -2*x(Co+i)*segment.rad(par)^2*cphi*x(Co+i)/(x(Co+par)*segment.rad(i)^2)))/x(Co+i);
        k=k+1;
    else % no junction
        Rart(i) = 8*prm.mu*segment.len(i)/(pi*segment.rad(i)^4)...
            *(1-3*(x(segint)/2 + x(segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.ha*prm.E));
        k=k+1;
    end
end
% Vein 
for i = 1 : vsegment.n
    segint = vsegment.init(i);
    segend = vsegment.end(i);
    if vnode.junction(segint)
        par = vsegment.num(vsegment.end==segint);
        cphi = cos(3*(vsegment.Gtheta(i)-vsegment.Gtheta(par))/4);
        Rvein(i) = (8*prm.mu*vsegment.len(i)*x(Cv+i)/(pi*vsegment.rad(i)^4)...
            *(1-3*(x(node.n+segint)/2 + x(node.n+segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.hv*prm.E))...
            + prm.Rho*x(Cv+par)^2/(2*pi^2*vsegment.rad(par)^4)...
            *(1+x(Cv+i)^2*vsegment.rad(par)^4/(x(Cv+par)^2*vsegment.rad(i)^4)...
            -2*x(Cv+i)*vsegment.rad(par)^2*cphi*x(Cv+i)/(x(Cv+par)*vsegment.rad(i)^2)))/x(Cv+i);
        k=k+1;
    else % no junction
        Rvein(i) = 8*prm.mu*vsegment.len(i)/(pi*vsegment.rad(i)^4)...
            *(1-3*(x(node.n+segint)/2 + x(node.n+segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.hv*prm.E));
        k=k+1;
    end
end
end
