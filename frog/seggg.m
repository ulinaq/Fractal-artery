function [node,segment,vnode,vsegment] = seggg(ra,rv)
load('frog_trees.mat');

[node,segment] = segmentasi(tree_a{1},ra,0);
[vnode,vsegment] = segmentasi(tree_v{1},rv,1);

function [node,segment] = segmentasi(tree,lim,type)
Ar = tree.segment.avrad > lim;
locA.conn=tree.segment.nodeconn(Ar);
locA.avr=tree.segment.avrad(Ar);
locA.L=tree.segment.L(Ar);
locA.c=tree.segment.c(Ar);
ac=cell2mat(locA.conn);
% delete unwanted segment vein
if type
    ac((ac(:,1)==269),:)=[];
    ac((ac(:,1)==250),:)=[];
    ac((ac(:,1)==211),:)=[];
    ac((ac(:,1)==218),:)=[];
    ac((ac(:,1)==176),:)=[];
    ac((ac(:,1)==212),:)=[];
    ac((ac(:,1)==220),:)=[];
    ac((ac(:,1)==201),:)=[];
    ac((ac(:,1)==188),:)=[];
    ac((ac(:,1)==190),:)=[];
    ac((ac(:,1)==201),:)=[];
    ac((ac(:,1)==188),:)=[];
    ac((ac(:,1)==188),:)=[];
    ac((ac(:,1)==190),:)=[];
else
    ac((ac(:,1)==121),:)=[];
end
ab=zeros(size(ac));
[b,m1,n1] = unique(ac,'first');
[c1,d1] =sort(m1);
for i=1:length(b)
    ab(ac==b(i))=i;
end

ab=reshape(ab,size(ac));
segment.init = ab(:,2);
segment.end = ab(:,1);
segment.len = locA.L;
segment.rad = locA.avr;
segment.n=length(ab);

node.avc=tree.node.avc(b,:);
node.XData=tree.node.avc(b,1);
node.YData=tree.node.avc(b,2);
node.isterminal=zeros(length(b),1);
node.junction=zeros(length(b),1);
for i=1:length(ac)
    ad=find(ac(:,2)==ac(i,1));
    if nnz(ad)>=2
        node.junction(ab(i,1))=1;
        node.Parent(ab(ad,2))=1;
    else
        node.junction(ab(i,1))=0;
    end
    if isempty(ad)
        node.isterminal(ab(i,1))=1;
    end
end
node.isroot=tree.node.isroot(b,:);
node.n=length(b);
node.isterminal=logical(node.isterminal);

% Compute the angle on bifurcation
end
end