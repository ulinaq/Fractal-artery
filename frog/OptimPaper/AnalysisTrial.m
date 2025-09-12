name = ['Optimization2'];
prm = Frog_parameter();
[Trans,Grid] = FO_Darcy(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[Onode,Osegment,Ovnode,Ovsegment] = segOne(prm,Grid,0,0);
[Xr,Grid] = FO_Capillary(Onode,Ovnode,Grid);
x = Frog_LinEq(prm,Onode,Osegment,Ovnode,Ovsegment,Xr.P);
Co = Onode.n + Ovnode.n;
Cv = Co + Osegment.n;
in = x([Co+Osegment.num(Osegment.isterminal)]);
out = x([Cv+Ovsegment.num(Ovsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);

% add flow rate as weight on graph plot
GVo=graph(Ovsegment.init,Ovsegment.end,abs(x(Cv+1:Cv+Ovsegment.n)));
GAo=graph(Osegment.init,Osegment.end,abs(x(Co+1:Cv)));
GVo.Edges.width=10*sqrt(abs(GVo.Edges.Weight)/sum(out));
GAo.Edges.width=10*sqrt(abs(GAo.Edges.Weight)/sum(in));

load('init.mat');
[Xr,Grid] = FO_Capillary(node,vnode,Grid);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);

% add flow rate as weight on graph plot
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+1:Cv+vsegment.n)));
GA=graph(segment.init,segment.end,abs(x(Co+1:Cv)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));

value = var2;
n = length(value);
%Update structure by x perturbation
[OptimNode,Optimsegment]=FO_update(value(1:round(n/4)), value(round(n/4)+1:round(n/2)),node,segment,Grid);
[OptimVnode,Optimvsegment]=FO_update(value(round(n/2)+1:round(3*n/4)), value(round(3*n/4)+1:end),vnode,vsegment,Grid);

[Xr,Grid] = FO_Capillary(OptimNode,OptimVnode,Grid); 
% Solve the linear system
x = Frog_LinEq(prm,OptimNode,Optimsegment,OptimVnode,Optimvsegment,Xr.P);
% Calculate pressure distribution in Darcy domain
Co = OptimNode.n + OptimVnode.n;  Cv = Co + Optimsegment.n;
in = x([Co+Optimsegment.num(Optimsegment.isterminal)]);
out = x([Cv+Optimvsegment.num(Optimvsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);

% Visualization
GVn=graph(Optimvsegment.init,Optimvsegment.end,abs(x(Cv+Optimvsegment.num)));
GAn=graph(Optimsegment.init,Optimsegment.end,abs(x(Co+Optimsegment.num)));
GVn.Edges.width=10*sqrt(abs(GVn.Edges.Weight)/sum(out));
GAn.Edges.width=10*sqrt(abs(GAn.Edges.Weight)/sum(in));
V = Flux0(Pa,Grid,Trans,q); 

figure('position', [0 0 600 800])
% clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pa*prm.mmHg,'AlphaData',Grid.B); axis ij; title(['Arterial vessel in ',name]); daspect([1 1 1]);
caxis([-10, 5.8166956273219]); set(gca,'color','w');
% colormap(gray);
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GA,'k','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width,'EdgeAlpha',0.3);
plot(GAo,'g','YData',Onode.XData,'XData',Onode.YData,'LineWidth',GAo.Edges.width,'EdgeAlpha',0.3);
plot(GAn,'r','YData',OptimNode.XData,'XData',OptimNode.YData,'LineWidth',GAn.Edges.width,'EdgeAlpha',0.3);
daspect([1 1 1])
set(gcf, 'Color', 'w'); 
export_fig(['Fig/new/', name, ' artery'],'-r300'); 

figure('position', [0 0 600 800])
% clf;
imagesc([1 515]*prm.pixel,[1 634]*prm.pixel,Pa*prm.mmHg,'AlphaData',Grid.B); axis ij; title(['Vein vessel in ',name]); daspect([1 1 1]);
caxis([0, 10.8166956273219]); set(gca,'color','w');
colormap(gray);
xlabel('x [mm]')
ylabel('y [mm]')
hold on
plot(GV,'k','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width);
plot(GVo,'g','YData',Ovnode.XData,'XData',Ovnode.YData,'LineWidth',GVo.Edges.width);
plot(GVn,'b','YData',OptimVnode.XData,'XData',OptimVnode.YData,'LineWidth',GVn.Edges.width);
daspect([1 1 1])
set(gcf, 'Color', 'w'); 
export_fig(['Fig/new/', name, ' vein'],'-r300');

resultA = get_terminal_difference(Onode, node, OptimNode, [name, 'Artery']);
resultV = get_terminal_difference(Ovnode, vnode, OptimVnode, [name, 'vein']);

function result = get_terminal_difference(TargetNode, InitNode, OptNode, name)
    x_target = TargetNode.XData - InitNode.XData;
    y_target = TargetNode.YData - InitNode.YData;
    x_opt = OptNode.XData - InitNode.XData;
    y_opt = OptNode.YData - InitNode.YData;
    d_target = sqrt(x_target.^2 + y_target.^2);
    d_opt = sqrt(x_opt.^2 + y_opt.^2);
    proj = (x_target.*x_opt + y_target.*y_opt)./d_target;
    dev = sqrt(d_opt.^2 - proj.^2);
    norm_proj = proj./d_target;
    norm_dev = dev./d_target;
    result.depth = InitNode.depth(InitNode.isterminal);
    result.proj = norm_proj(InitNode.isterminal);
    result.dev = norm_dev(InitNode.isterminal);
    ind = 1;
    for i = min(result.depth): max(result.depth)
        xi(ind) = i;
        n = length(result.dev(result.depth==i));
        devAvg(ind) = sum(result.dev(result.depth==i))/n;
        projAvg(ind) = sum(result.proj(result.depth==i))/n;
        ind = ind +1;
    end
    figure();
    scatter(xi, devAvg);title(name);
    hold on
    scatter(xi, projAvg, '*');
end