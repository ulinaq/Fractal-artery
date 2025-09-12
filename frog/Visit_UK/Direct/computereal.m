Reps = 80;
for iter = 1:20
Roi = CircleROI(xcell,ycell,Reps,Grid);
roi = full(mean(coS(Roi,:)))'; maxR = max(roi);
air = full(def_AIF(coS(:,:),segment))'; maxA = max(air);
[Pfit,con, resnorm] = funfit(Time, air, roi, 1);
figure();plot(Time,con,Time,roi)
Fl1 = Pfit.BF;
[Pfit,con, resnorm] = funfit(Time, air, roi, 2);
figure();plot(Time,con,Time,roi)
Fl2 = Pfit.BF;
[Pfit,con, resnorm] = funfit(Time, air, roi, 3);
figure();plot(Time,con,Time,roi)
Fl3 = Pfit.BF;
[Pfit,con, resnorm] = funfit(Time, air, roi, 4);
figure();plot(Time,con,Time,roi)
Fl4 = Pfit.BF;

Co = node.n + vnode.n; Cv = Co+segment.n; Cq = Cv+vsegment.n;
Skel = zeros(Grid.Nx,Grid.Ny);        
nm = 1;        
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    Skel(segment.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    Skel(vsegment.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end

for i=1:Connect.n
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    Skel(Connect.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end
for i=1:node.n
    Skel(node.ind{i}) = nm;
    nm = nm + 1; 
end
for i=1:vnode.n
    Skel(vnode.ind{i})= nm;
    nm = nm + 1; 
end

%% Compute the total flow through vessels into the ROI
area = Skel(Roi);
area = unique(nonzeros(area));

sgde = area(area<=segment.n);
vgde = area(area>segment.n & area<=segment.n+vsegment.n) - segment.n;
cgde = area(area>segment.n+vsegment.n & area<=segment.n+vsegment.n+Connect.n) - (segment.n+vsegment.n);
nde = area(area>segment.n+vsegment.n+Connect.n & area<=node.n+segment.n+vsegment.n+Connect.n) - (segment.n+vsegment.n+Connect.n);
vde = area(area>node.n+segment.n+vsegment.n+Connect.n) - (node.n+segment.n+vsegment.n+Connect.n);

% remove unwanted terminal connection
if ~isempty(cgde)
    conA = Connect.AV(cgde,1);
    conV = Connect.AV(cgde,2);
    tmp = cgde;
    for i = 1 : length(cgde)
        idx = ismember(nde,conA(i));
        idy = ismember(vde,conV(i));
%         if any(idx) && x(Cq+cgde(i))<0 || any(idy) && x(Cq+cgde(i))>0
%             cgde;
%         else 
        if any(idx) && any(idy)
            tmp(tmp==cgde(i)) = [];
        elseif any(idx) && x(Cq+cgde(i))>0
            tmp(tmp==cgde(i)) = [];
        elseif any(idy) && x(Cq+cgde(i))<0
            tmp(tmp==cgde(i)) = [];
        end
    end
    cgde = tmp;
end

% remove unwanted vein segment
if ~isempty(nde)
    clog = node.clog(nde);
    clog = nde(clog);
    for i = 1 : length(clog)    % remove clogged vessels
        ab = segment.num(segment.end==clog(i));
        ba = segment.num(segment.init==clog(i));
        idx = ismember(sgde,[ba,ab]);
        sgde(idx)=[];
    end
    junc = node.isterminal(nde);
    nde = nde(~junc);
    for i = 1 : length(nde)     % remove connected and junction vessel
        ba = segment.num(segment.init==nde(i));
        ab = segment.num(segment.end==nde(i));
        % add some missing segments around node
        sgde = unique([sgde;ab';ba']);
    end
    for i = 1 : length(nde)     % remove connected and junction vessel
        ba = segment.num(segment.init==nde(i));
        ab = segment.num(segment.end==nde(i));
        if x(Co+ab)>0   % remove unwanted segment
            ba = ba(x(Co+ba)>0);
            idx = ismember(sgde,ba);
            sgde(idx)=[];
        else
            ba = ba(x(Co+ba)>0);
            idx = ismember(sgde,[ba,ab]);
            sgde(idx)=[];
        end
    end
end

% remove unwanted venous segment
if ~isempty(vde)
    clog = vnode.clog(vde);
    clog = vde(clog);
    for i = 1 : length(clog)    % remove clogged vessel
        ab = segment.num(segment.end==clog(i));
        ba = segment.num(segment.init==clog(i));
        idx = ismember(vgde,[ba,ab]);
        vgde(idx)=[];
    end
    junc = vnode.isterminal(vde);
    tde = vde(~junc);
    for i = 1 : length(tde)     % remove connected and junction vessel
        ba = vsegment.num(vsegment.init==tde(i));
        ab = vsegment.num(vsegment.end==tde(i));
        % add some missing segments around node
        vgde = unique([vgde;ab';ba']);
    end
    for i = 1 : length(tde)     % remove connected and junction vessel
        ba = vsegment.num(vsegment.init==tde(i));
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)<0
            ba = ba(x(Cv+ba)<0);
            idx = ismember(vgde,ba);
            vgde(idx)=[];
        else
            ba = ba(x(Cv+ba)<0);
            idx = ismember(vgde,[ba,ab]);
            vgde(idx)=[];
        end
    end
    junc = vnode.isterminal(vde);
    tde = vde(junc);
    for i = 1 : length(tde)
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)>0
            idx = ismember(vgde,ab);
            vgde(idx)=[];        
        end
    end
end

% Radius
% Radi = [segment.rad(sgde);vsegment.rad(vgde);Connect.rad(cgde)'];
% Total flow into ROI
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*60/100/Grid.volume;
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*60/100/(prm.pixel^2)/0.55;
Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*60/10/prm.pixel;
% Fl = sum(abs(x([Co+sgde;Cq+cgde])))*0.55*600/100/prm.pixel;
% creat a random located circle ROI with a predetermined radius
Flsa(iter,:) = [Fl,Fl1,Fl2,Fl3,Fl4];

figure()
imagesc(maXc,[0 100]);colormap(bone);axis image;axis off; title(['ROI (rad=',num2str(Reps),')']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
plot(GV,'c','YData',vnode.XData/prm.pixel,'XData',vnode.YData/prm.pixel,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
daspect([1 1 1])
plot(GA,'r','YData',node.XData/prm.pixel,'XData',node.YData/prm.pixel,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1])
end
    function AIF = def_AIF(c,segment)
    [a,b]=maxk(segment.rad,4);
    reg = segment.distrind(b);
    as=[];
    for i = 1:length(reg)
        as = [as;cell2mat(reg{i})];
    end
    reg = unique(as);
    AIF = mean(c(reg,:));    
    end
    
    function area = CircleROI(xcell,ycell,Rep,Grid)
        a = 1.5;
        Reps = a*Rep;
        Rx = randi([Reps Grid.Nx-Reps]);
        Ry = randi([Reps Grid.Ny-Reps]);
        Center = sub2ind([Grid.Nx Grid.Ny],Rx,Ry);
        area=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Rep^2;
    end

function Fl = Floreal(prm,Grid,node,vnode,segment,vsegment,Connect,x,Roi)
Co = node.n + vnode.n; Cv = Co+segment.n; Cq = Cv+vsegment.n;
Skel = zeros(Grid.Nx,Grid.Ny);        
nm = 1;        
for i=1:node.n
    Skel(node.ind{i}) = nm;
    nm = nm + 1; 
end
for i=1:vnode.n
    Skel(vnode.ind{i})= nm;
    nm = nm + 1; 
end
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    Skel(segment.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    Skel(vsegment.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end

for i=1:Connect.n
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    Skel(Connect.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end
%% Compute the total flow through vessels into the ROI
area = Skel(Roi);
area = unique(nonzeros(area));

nde = area(area<=node.n);
vde = area(area>node.n & area<=node.n+vnode.n) - node.n;
sgde = area(area>node.n+vnode.n & area<=node.n+vnode.n+segment.n) - (node.n+vnode.n);
vgde = area(area>node.n+vnode.n+segment.n & area<=node.n+vnode.n+segment.n+vsegment.n) - (node.n+vnode.n+segment.n);
cgde = area(area>node.n+vnode.n+segment.n+vsegment.n) - (node.n+vnode.n+segment.n+vsegment.n);

% remove unwanted arterial segment
if ~isempty(nde)
    junc = node.junction(nde);
    nde = nde(junc);
    for i = 1 : length(nde)
        ba = segment.num(segment.init==nde(i));
        ab = segment.num(segment.end==nde(i));
        if x(Co+ab)>0
            ba = ba(x(Co+ba)>0);
            idx = ismember(sgde,ba);
            sgde(idx)=[];
        else
            ba = ba(x(Co+ba)>0);
            idx = ismember(sgde,[ba,ab]);
            sgde(idx)=[];
        end
    end
end

% remove unwanted venous segment
if ~isempty(vde)
    junc = vnode.junction(vde);
    tde = vde(junc);
    for i = 1 : length(tde)
        ba = vsegment.num(vsegment.init==tde(i));
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)<0
            ba = ba(x(Cv+ba)<0);
            idx = ismember(vgde,ba);
            vgde(idx)=[];
        else
            ba = ba(x(Cv+ba)<0);
            idx = ismember(vgde,[ba,ab]);
            vgde(idx)=[];
        end
    end
    junc = vnode.isterminal(vde);
    tde = vde(junc);
    for i = 1 : length(tde)
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)>0
            idx = ismember(vgde,ab);
            vgde(idx)=[];        
        end
    end
end

% remove unwanted terminal connection
if ~isempty(cgde)
    conA = Connect.AV(cgde,1);
    conV = Connect.AV(cgde,2);
    tmp = cgde;
    for i = 1 : length(cgde)
        idx = ismember(nde,conA(i));
        idy = ismember(vde,conV(i));
        if any(idx) && x(Cq+cgde(i))<0 || any(idy) && x(Cq+cgde(i))>0
            cgde;
        else 
            tmp(tmp==cgde(i)) = [];
        end
    end
    cgde = tmp;
end
% Radius
% Radi = [segment.rad(sgde);vsegment.rad(vgde);Connect.rad(cgde)'];
% Total flow into ROI
Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*60/100/Grid.volume;
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*60/100/(prm.pixel^2)/0.55;
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*60/10/prm.pixel/0.55;
% Fl = sum(abs(x([Co+sgde;Cq+cgde])))*0.55*600/100/prm.pixel;
end