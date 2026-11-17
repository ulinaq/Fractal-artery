function [node,segment,vnode,vsegment] = Tree_edit(ra,rv)
load('frog_trees.mat');
[node,segment] = seggg(tree_a{1},ra);
[vnode,vsegment] = seggg(tree_v{1},rv);
% numbering
node.num=[1:node.n]';   segment.num=[1:segment.n]';
vnode.num=[1:vnode.n]'; vsegment.num=[1:vsegment.n]';
% Node exist or not? after deleting node procedure
node.isexist=logical(ones(node.n,1));
vnode.isexist=logical(ones(vnode.n,1));
% Segment stil exist?
segment.isexist=logical(ones(segment.n,1));
vsegment.isexist=logical(ones(vsegment.n,1));

%% tambahan (nanti) ra>0.65, rv>0.95
if ra==0.65
    vsegment.init([end+1,end+2])=[7,9];
    vsegment.end([end+1,end+2])=[6,8];
    vsegment.len([end+1,end+2])=vsegment.len(vsegment.end==7);
    vsegment.rad([end+1,end+2])=vsegment.rad(vsegment.end==7);
    vsegment.n=length(vsegment.len);
    vnode.junction(7)=1;
    vnode.isterminal([9,7])=0;
    % connecting parallel
    [vnode,vsegment]=connecting(vnode,vsegment,47,48)

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
%% plot
A = imread('frog_vein_low2.png');
B = im2bw(A);
A = 1-B;
c = find(B);
Bz = size(B);
GV=graph(vsegment.init,vsegment.end,vsegment.len);
GA=graph(segment.init,segment.end,segment.len);

figure()
imshow(A)
hold on;
plot(GV,'YData',vnode.XData,'XData',vnode.YData);
daspect([1 1 1])
% plot(GA,'YData',node.XData,'XData',node.YData);
% title('Artery Tree');
% daspect([1 1 1])

%% note segment root and terminal
segment = segroot(node,segment);
vsegment = segroot(vnode,vsegment);

% delete segment based on node
function [node,segment]= del_segment(node,segment,del)
    segment.isexist(del)=0;
    loc=segment.num(segment.end==del);
    awal=segment.init(loc);
    akhir=segment.end(loc);
    if node.isterminal(akhir)
        node.isexist(akhir)=0;
    else
    end
    if node.junction(awal)
        if nnz(segment.end==awal)>2 % three branches
        else % two branches
            node.junction(awal)=0;
        end
    else
    end
            
    node.isterminal(akhir)=0;
    node.isroot(awal)=1;
end
% connecting parallel
function [node,segment]=connecting(node,segment,awal,akhir)
    segment.init(end+1)=awal;
    segment.end(end+1)=akhir;
    segment.rad(end+1)=segment.rad(segment.end==akhir);
    segment.len(end+1)=sqrt((node.XData(awal)-node.XData(akhir))^2+(node.YData(awal)-node.YData(akhir))^2);
    segment.isexist(end+1)=1;
    node.isterminal(akhir)=0;
    node.isroot(awal)=1;
end
end