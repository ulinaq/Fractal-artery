% function [node,Grid] = FS_Capillary(node,vnode,segment,vsegment,Grid)
RepsA = 30; RepsV = 30;   % Epsilon radii for mollifier around terminal nodes
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz);
ain = node.num(node.isterminal);
vout = vnode.num(vnode.isterminal);
rmin = min(segment.rad); j = 0;
Voe = vout;
for i=1:length(node.input)
    % Create circle region for spreading blood region
    Center = node.input(i);
    Rc = RepsA*(segment.rad(segment.end==ain(i))/rmin)^2;
    [bw1,bw2] = moll(xcell,ycell,zcell,Rc,Grid,Center); 
    Srdg(i) = any(ismember(vnode.output,Grid.num(bw1)));
    if Srdg(i) % The nearest venous connections from an arterial terminal
        j=j+1;
        Conn{j} = vout(ismember(vnode.output,Grid.num(bw1)));
        Voe = Voe(~ismember(Voe,Conn{j}));
    else
        node.clog(ain(i)) = true;      % disconnected terminals
    end
end
vnode.clog(Voe) = true; % disconnected terminals

Connect.diverging = ain(Srdg);
Connect.AV = [];
for i = 1 : j
    nn = cellfun('size',Conn(i),2);
    Connect.AV(end+1:end+nn,:) = [ones(nn,1)*Connect.diverging(i),Conn{i}'];
end
Connect.converging = unique(Connect.AV(:,2));
kl = size(Connect.AV,1);
Connect.n = kl; j=1;
for i = 1 : kl
    awal = Connect.AV(i,1);     akhir = Connect.AV(i,2);
    Connect.len(i) = sqrt((node.XData(awal)-vnode.XData(akhir))^2+(node.YData(awal)-vnode.YData(akhir))^2);
    Connect.rad(i) = vsegment.rad(vsegment.end==akhir);
    point1 = node.avc(awal,:);
    point2 = vnode.avc(akhir,:);
    t = 0:1/norm(point1-point2):1;
    Cor = round(repmat(point1,length(t),1)'+(point2-point1)'*t);
    Connect.c{i} = Cor';
    Connect.ind{i} = double(sub2ind([Grid.Nx Grid.Ny],Cor(1,:),Cor(2,:))');
    Connect.distrind{i} = num2cell(Connect.ind{i});
    Connect.distrvol{i} = round(ones(length(Cor),1)*Connect.rad(i)/min(Connect.rad))';
    Connect.Cnum{i} = (j:j-1+cellfun('size',Connect.c(i),1))';
    j = j + cellfun('length',Connect.c(i));
end
    Connect.ind{Connect.n} = sub2ind([Grid.Nx Grid.Ny],Connect.c{Connect.n}(:,1),Connect.c{Connect.n}(:,2));
    Connect.distrind{Connect.n} = num2cell(Connect.ind{Connect.n});

function [bw1,bw2] = moll(xcell,ycell,zcell,Reps,Grid,Center)
        bw1=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Reps^2;
        bw2=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=8*Reps^2;
end
% end