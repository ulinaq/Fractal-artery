function [xN,Pnew,rad]=FO_Rad(x,node,segment,vnode,vsegment,Pa,Pv,Grid,Xr,prm,Trans,V)
for i=1:segment.n
    rold = segment.rad(i);
    h = 0.3*rold;
    E = 70;
    if isempty(segment.c{i})
        mid = node.avc(segment.init(i),:);
    else
        mid = segment.c{i}(ceil(length(segment.c{i})/2),:);
    end
    Pext = Pa(sub2ind([Grid.Nx Grid.Ny],mid(1),mid(2)));
    Pin = (x(segment.end(i))+x(segment.init(i)))/2;
    segment.rad(i) = rold*h*E/(h*E-0.75*(Pin-Pext)*rold);
end
for i=1:vsegment.n
    rold = vsegment.rad(i);
    h = 0.1*rold;
    E = 40;
    if isempty(vsegment.c{i})
        mid = vnode.avc(vsegment.init(i),:);
    else
        mid = vsegment.c{i}(floor(length(vsegment.c{i})/2),:);
    end
    Pext = Pv(sub2ind([Grid.Nx Grid.Ny],mid(1),mid(2)));
    Pin = (x(vsegment.end(i))+x(vsegment.init(i)))/2;
    vsegment.rad(i) = rold*h*E/(h*E-0.75*(Pin-Pext)*rold);
end
rad.art = segment.rad;
rad.vein = vsegment.rad;
xN = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = xN([Co+segment.num(segment.isterminal)]);
out = xN([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+xN(end);
Pnew = P;
Pan = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pvn = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
GV=graph(vsegment.init,vsegment.end,abs(xN(Cv+vsegment.num)));
GA=graph(segment.init,segment.end,abs(xN(Co+segment.num)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
Vn = Flux(prm,Pan,Pvn,Grid,Trans,q,node,vnode);
% Compute the difference
B = Grid.B;
rad.dPa=norm(Pa(B)-Pan(B)); rad.dPv=norm(Pv(B)-Pvn(B));
rad.dTa=norm(V.Ta(B)-Vn.Ta(B)); rad.dTv=norm(V.Tv(B)-Vn.Tv(B));


B = Grid.B;
bb = 1-B;
PP = ones(size(B))*(min(P)-1);
PP(B) = Pan(B);
figure()
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,PP); axis ij; title('Pressure in artery compartment'); daspect([1 1 1])
caxis([min(P) max(P)])
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'P [kPa]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
% plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])

% PP = ones(size(B))*min(Pv(B));
PP(B) = Pvn(B);
figure()
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,PP);  title('Pressure in vein compartment');daspect([1 1 1])
caxis([min(P) max(P)])
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar;
title(hcb,'P [kPa]')
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
% plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
% daspect([1 1 1])

figure()
todisp = Vn.Ta; todisp(logical(1-Grid.B))=1.01*max(Vn.Ta(B));
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,todisp); title('Time of flight in artery compartment');
daspect([1 1 1]); caxis([0 1.01*max(Vn.Ta(B))])
hold on
hcb=colorbar
title(hcb,'T [s]')
xlabel('x [mm]')
ylabel('y [mm]')
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])

figure()
todisp = Vn.Tv; todisp(logical(1-Grid.B))=1.01*max(Vn.Tv(B));
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,todisp); title('Time of flight in vein compartment');
daspect([1 1 1]); caxis([0 1.01*max(Vn.Tv(B))])
hold on
hcb=colorbar
title(hcb,'T [s]')
xlabel('x [mm]')
ylabel('y [mm]')
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
end