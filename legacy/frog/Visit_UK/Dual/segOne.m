function [node,segment,vnode,vsegment] = segOne(prm,Grid,ra,rv)
load('frog_trees.mat');

[node,segment] = segmentasi(tree_a{1},ra,0);
[vnode,vsegment] = segmentasi(tree_v{1},rv,1);
%% tambahan (nanti) ra>0.65, rv>0.95
if ra==0.65
    vsegment.init([end+1,end+2])=[7,9];
    vsegment.end([end+1,end+2])=[6,8];
    vsegment.len([end+1,end+2])=vsegment.len(vsegment.end==7);
    vsegment.rad([end+1,end+2])=vsegment.rad(vsegment.end==7);
    vsegment.n=length(vsegment.len);
    vnode.junction(9)=1;
    vnode.isterminal([9,7,46])=0;
    % connecting parallel
    [vnode,vsegment]=connecting(vnode,vsegment,45,46);
    vsegment.num=1:vsegment.n;
    vsegment.isterminal(end+1:1:end+3)=0;
    vsegment.isroot(end+1:1:end+3)=0;
    vsegment.c(end+1:1:end+3)={[]};
    vsegment = segroot(vnode,vsegment);
    segment = segroot(node,segment);
end
%% ra>1 rv>2
if ra==1
    segment.init(end+1)=10;
    node.isterminal(10)=0;
    segment.end(end+1)=4;
    segment.len(end+1)=segment.len(segment.end==10);
    segment.rad(end+1)=segment.rad(segment.end==10);
    segment.n=length(segment.len);
end
% connecting parallel
%% Change dimension to real dimension
node.XData = node.XData*prm.pixel;
node.YData = node.YData*prm.pixel;
vnode.XData = vnode.XData*prm.pixel;
vnode.YData = vnode.YData*prm.pixel;
segment.len = segment.len*prm.pixel;
segment.rad = segment.rad*prm.pixel;
vsegment.len = vsegment.len*prm.pixel;
vsegment.rad = vsegment.rad*prm.pixel;
% locate mid point of segment on capillary bed
segment.ext = zeros(segment.n,1);
for i=1:segment.n
    if isempty(segment.c{i})
        mid = node.avc(segment.init(i),:);
    elseif segment.isroot(i) % root segment, 
        k = segment.init(segment.end(i));
        mid = segment.c{k}(ceil(length(segment.c{k})/2),:);
    else
        mid = segment.c{i}(ceil(length(segment.c{i})/2),:);
    end
    segment.ext(i) = sub2ind([Grid.Nx Grid.Ny],mid(1),mid(2));
end
vsegment.ext = zeros(vsegment.n,1);
for i=1:vsegment.n
    if isempty(vsegment.c{i})
        mid = vnode.avc(vsegment.init(i),:);
%     elseif vsegment.isroot(i)
%         k = vsegment.init(vsegment.end(i));
%         mid = vsegment.c{k}(ceil(length(vsegment.c{k})/2),:);
    else
        mid = vsegment.c{i}(floor(length(vsegment.c{i})/2),:);
    end
    vsegment.ext(i) = sub2ind([Grid.Nx Grid.Ny],mid(1),mid(2));
end

% Compute angle
segment = Bif_angle(node,segment);
vsegment = Bif_angle(vnode,vsegment);

% Locate nodes in the capillary bed
X = ceil((node.XData(node.isterminal))./Grid.hx);
Y = ceil(node.YData(node.isterminal)./Grid.hy);
Xout = floor((vnode.XData(vnode.isterminal))./Grid.hx);
Yout = floor((vnode.YData(vnode.isterminal))./Grid.hy);

if max(Xout)>=Grid.Nx | max(Yout)>=Grid.Ny | min(Xout)<=0 | min(Yout)<=0 ...
        | max(X)>=Grid.Nx | max(Y)>=Grid.Ny | min(X)<=0 | min(Y)<=0
    state.P=NaN; state.pvec=NaN;
    return;
end
node.input = sub2ind([Grid.Nx Grid.Ny],X,Y);
vnode.output = sub2ind([Grid.Nx Grid.Ny],Xout,Yout);
segment = Locate(node,segment,Grid);
vsegment = Locate(vnode,vsegment,Grid);

segment.rad = 15*segment.rad;
vsegment.rad = 40*vsegment.rad;
node.clog = false(node.n,1);
vnode.clog = false(vnode.n,1);
function segment = Locate(node,segment,Grid)
    emp = segment.num(cellfun('isempty',segment.c));
    for i = 1 : length(emp)
    point1 = node.avc(segment.end(emp(i)),:);
    point2 = node.avc(segment.init(emp(i)),:);
    t = 0:1/norm(point1-point2):1;
    Cor = round(repmat(point1,length(t),1)'+(point2-point1)'*t);
    segment.c{emp(i)} = Cor';
    segment.ind{emp(i)} = double(sub2ind([Grid.Nx Grid.Ny],Cor(1,:),Cor(2,:))');
    segment.distrind{emp(i)} = num2cell(segment.ind{emp(i)});
    segment.distrvol{emp(i)} = round(ones(length(Cor),1)*segment.rad(emp(i))/min(segment.rad))';
    end
    segment.ind{segment.n} = sub2ind([Grid.Nx Grid.Ny],segment.c{segment.n}(:,1),segment.c{segment.n}(:,2));
    segment.distrind{segment.n} = num2cell(segment.ind{segment.n});
    segment.distrvol{segment.n} = round(ones(length(segment.c{segment.n}),1)*segment.rad(segment.n)/min(segment.rad))';
 
    j=1;
    for i = 1 : segment.n
        segment.Cnum{i} = (j:j-1+cellfun('size',segment.c(i),1))';
        j = j + cellfun('length',segment.c(i));
    end
end

function [node,segment]=connecting(node,segment,awal,akhir)
    segment.rad(end+1)=segment.rad(segment.end==akhir);
    segment.init(end+1)=awal;
    segment.end(end+1)=akhir;
    segment.len(end+1)=sqrt((node.XData(awal)-node.XData(akhir))^2+(node.YData(awal)-node.YData(akhir))^2);
    node.isterminal(akhir)=0;
    node.isroot(awal)=0;
    segment.n=length(segment.len);
end

function [node,segment] = segmentasi(tree,lim,type)
Ad = tree.segment.avrad > 0;
locA.conn=tree.segment.nodeconn(Ad);
ac=cell2mat(locA.conn);
segment.radius = tree.radius;
% delete unwanted segment vein
if type 
    del=[269,250,211,218,176,212,220,201,188,190,201,263];
    for i=1:length(del)
        Ad((ac(:,1)==del(i)))=0;
    end
    ac=ac(Ad,:);
else % artery
    del=[121,81];
    for i=1:length(del)
        Ad(ac(:,1)==del(i))=0;
    end
    ac=ac(Ad,:);
end
Ar = tree.segment.avrad > lim;
Ar(Ad==0) = 0;
locA.conn=tree.segment.nodeconn(Ar);
ac=cell2mat(locA.conn);
locA.avr=tree.segment.avrad(Ar);
locA.L=tree.segment.L(Ar);
locA.c=tree.segment.c(Ar);
ab=zeros(size(ac));
[b,m1,n1] = unique(ac,'first');
[n1,d1] =sort(m1);
for i=1:length(b)
    ab(ac==b(i))=i;
end

ab=reshape(ab,size(ac));
segment.init = ab(:,2);
segment.end = ab(:,1);
segment.len = locA.L;
segment.rad = locA.avr;
segment.c = locA.c;
segment.dist = tree.segment.dist(Ar);
segment.ind = tree.segment.ind(Ar);
segment.distrvol = tree.segment.distrvol(Ar);
segment.distrind = tree.segment.distrind(Ar);
segment.geodist = tree.segment.geodesdist(Ar);
segment.Cnum = tree.segment.num(Ar);

node.avc=tree.node.avc(b,:);
node.XData=tree.node.avc(b,1);
node.YData=tree.node.avc(b,2);
node.depth=tree.node.depth(b,:);
node.ind = tree.node.ind(b,:);
node.distrind = tree.node.distrind(b,:);
%% connect vein tree
if lim==0
if type  
    awal=198; akhir=205;
    ac(end+1,:)=[akhir,awal];
    segment.rad(end+1)=segment.rad(segment.end==akhir);
    segment.len(end+1)=sqrt((node.XData(awal)-node.XData(akhir))^2+(node.YData(awal)-node.YData(akhir))^2);
    segment.init(end+1)=awal;
    segment.end(end+1)=akhir;
    segment.c(end+1)= {[]};
    ab=zeros(size(ac));
[b,m1,n1] = unique(ac,'first');
[n1,d1] =sort(m1);
for i=1:length(b)
    ab(ac==b(i))=i;
end
ab=reshape(ab,size(ac));
else 
    awal=7; akhir=21; midsegment=4; othe=6; baru=max(ab(:))+1; bar=max(ac(:))+1;
    ac(4,:)=[];
    ac(end+1,:)=[akhir,bar];
    ac(end+1,:)=[othe,bar];
    ac(end+1,:)=[bar,awal];
    segment.rad(end+1)=segment.rad(segment.end==akhir);
    segment.len(end+1)=sqrt((segment.c{4}(118,1)-node.XData(akhir))^2+(segment.c{4}(118,2)-node.YData(akhir))^2);
    segment.init(end+1)=baru;
    segment.end(end+1)=akhir;
    segment.c(end+1)={[]};
    segment.rad(end+1)=segment.rad(4);
    segment.len(end+1)=sqrt((segment.c{4}(118,1)-node.XData(awal))^2+(segment.c{4}(118,2)-node.YData(awal))^2);
    segment.init(end+1)=awal;
    segment.end(end+1)=baru;
    segment.c(end+1)={segment.c{4}(118:end,:)};
    segment.len(4)=118/135*segment.len(4);
    segment.init(4)=baru;
    segment.c(4)={segment.c{4}(1:118,:)};
    segment.n=length(segment.len);
    node.avc(baru,:)=segment.c{4}(end,:);
    node.XData(baru)=segment.c{4}(end,1);
    node.YData(baru)=segment.c{4}(end,2);
    node.ind{baru}=segment.ind{4}(end-4:end);
    node.depth(baru)=node.depth(othe);
    node.depth(othe)=node.depth(othe)+1;
    ab=zeros(size(ac));
[b,m1,n1] = unique(ac,'first');
[n1,d1] =sort(m1);
for i=1:length(b)
    ab(ac==b(i))=i;
end
ab=reshape(ab,size(ac));
end
end
%%
node.isterminal=zeros(length(b),1);
node.junction=zeros(length(b),1);
for i=1:length(ac)
    ad=find(ac(:,2)==ac(i,1));
    if nnz(ad)>=2
        node.junction(ab(i,1))=1;
        node.Parent(ab(ad,2))=1;
        % Compute the angle
    else
        node.junction(ab(i,1))=0;
    end
    if isempty(ad)
        node.isterminal(ab(i,1))=1;
    end
end
if type & lim==0
    node.isterminal(205)=0;
else
    b(end)=b(end-1);
    node.isterminal(21)=0;
end
node.n=length(b);
node.isroot=tree.node.isroot(b,:);
node.num=1:node.n;
node.isterminal=logical(node.isterminal);
segment.n=length(segment.len);
segment.num=1:segment.n;

% note segment root and terminal
segment = segroot(node,segment);

% Clear artifact / big terminal
if max(segment.rad(segment.isterminal))>1.5*mean(segment.rad(segment.isterminal))
    segment.rad(segment.rad==max(segment.rad(segment.isterminal)))=1.5*mean(segment.rad(segment.isterminal));
end
end
% note segment root and terminal
function segment = segroot(node,segment)
root = find(node.isroot);
rot=[];
for k=1:length(root)
    rot=[rot,segment.num(segment.init==root(k))];
end
segment.isroot = logical(zeros(segment.n,1));
segment.isroot(rot) = 1;
root = find(node.isterminal);
rot=[];
for k=1:length(root)
    rot=[rot,segment.num(segment.end==root(k))];
end
segment.isterminal = logical(zeros(segment.n,1));
segment.isterminal(rot) = 1;
end
function [segment] = Bif_angle(node,segment)
for i = 1:segment.n
    % The global angle between two nodes respect to x-axis (2D)
    segment.Gtheta(i)= angle(node.avc(segment.init(i),:),node.avc(segment.end(i),:));
    % The local angle of a segment close to node
    if isempty(segment.c{i})
        segment.Ltheta(i)=segment.Gtheta(i);
    elseif min(size(segment.c{i}))>4
        segment.Ltheta(i)= angle(segment.c{i}(end,:),segment.c{i}(end-3,:));
    elseif min(size(segment.c{i}))>2
        segment.Ltheta(i)= angle(segment.c{i}(end,:),segment.c{i}(end-1,:));
    else
        segment.Ltheta(i)= segment.Gtheta(i);
    end
     
end
end
% angle of segment respect to x-axis (2D)
function ksi = angle(Pinit,Pend) 
    a=sqrt((Pinit(1)-Pend(1))^2+(Pinit(2)-Pend(2))^2);   % scale factor
    b=1;
    c=sqrt((Pinit(1)+1-Pend(1))^2+(Pinit(2)-Pend(2))^2);
    alpha=acos((a^2+b^2-c^2)/(2*a*b));
    if Pend(2)>=Pinit(2)
        ksi=alpha;
    else
        ksi=-alpha; 
    end
end

end