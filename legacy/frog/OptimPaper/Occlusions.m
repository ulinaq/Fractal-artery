%%%% OCCLUSION EXPERIMENTS %%%%
% 
function Occl = Occlusions(type)
prm = Frog_parameter();
prm.type = 1;
[Trans,Grid] = FO_Darcy3(prm);    % Load the image domain and create darcy domain. Then compute transmibility
% ResT = zeros(4,20); 
% QT = zeros(4,20); 
xSave = zeros(765,4,20);
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
[Xr,Grid] = FO_Capillary2(node,vnode,Grid);
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
if type == 1
    guess = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
    x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,guess);
else
    x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
end
Co = node.n + vnode.n;  Cv = Co + segment.n;
RiniT= (prm.Pin-prm.Pout)/sum(x([Co+segment.num(segment.isroot)]));
% The occlusion vessel percentage
for jk = 1:4
for jn = 1:20
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
if jn==20
    switch jk
        case 1
            segment.rad(74)=segment.rad(74)/100;   % segment that is damaged has much smaller radii
        case 2
            segment.rad(85)=segment.rad(85)/100;   % segment that is damaged has much smaller radii
        case 3
            vsegment.rad(223)=vsegment.rad(223)/100;
        case 4
            vsegment.rad(173)=vsegment.rad(173)/100;
    end
else
    switch jk
        case 1
            segment.rad(74)=(20-jn)*segment.rad(74)/20;   % segment that is damaged has much smaller radii
        case 2
            segment.rad(85)=(20-jn)*segment.rad(85)/20;   % segment that is damaged has much smaller radii
        case 3
            vsegment.rad(223)=(20-jn)*vsegment.rad(223)/20;
        case 4
            vsegment.rad(173)=(20-jn)*vsegment.rad(173)/20;
    end
end
[Xr,Grid] = FO_Capillary2(node,vnode,Grid);
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);
if type == 1
    if jn ==1
        guess = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
    else
        guess = xSave(:,jk,jn-1);
    end
    x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,guess);
else
    x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
end
Co = node.n + vnode.n;  Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.C*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
xSave(:,jk,jn) = x;
Occl.QT(jk,jn) = sum(x([Co+segment.num(segment.isroot)])); 
Occl.ResT(jk,jn) = (prm.Pin-prm.Pout)/Occl.QT(jk,jn);
    dPa = mean(x(node.isterminal));
    dPv = mean(x(node.n+vnode.num(vnode.isterminal)));
    dPC = 0.5*(mean(Pv(Grid.B))+ mean(Pa(Grid.B)));
Occl.RAtree(jk,jn) = (prm.Pin-dPa)/(prm.Pin-prm.Pout)*100;
Occl.RAcap(jk,jn) = std(Pa(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*100;
Occl.RVcap(jk,jn) = std(Pv(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*100;
Occl.RVtree(jk,jn) = (dPv-prm.Pout)/(prm.Pin-prm.Pout)*100;
name = ['Occl2-',num2str(jk),'_on_',num2str(jn*5),'%'];
end
end

PP = [ones(1,4)*RiniT;ResT(:,1:20)'];
Fr = [0:5:100];
figure;plot(Fr,PP,'LineWidth',2)
if type == 1
    title('Total resistance in the nonlinear occlusions'); 
else
    title('Total resistance in the linear occlusions');     
end
xlabel('Vessel occlusion [%]')
ylabel('R [kg.mm^{-4}s^{-1}]')
legend('artery-1','artery-2','vein-1','vein-2')
end