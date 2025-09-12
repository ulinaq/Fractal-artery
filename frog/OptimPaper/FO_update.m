function [nod,segmen]=FO_update(x,tta,node,segment,Grid)
n=length(x);
% precentage of changing segment length
Len=zeros(segment.n,1);
for i=1:segment.n
    Len(i)=1+x(1+mod(i,n))*node.depth(segment.end(i))/max(node.depth);
end
distance=zeros(segment.n,1);
for i=1:segment.n
    distance(i)=Len(i)*sqrt((node.XData(segment.end(i))-node.XData(segment.init(i)))^2+(node.YData(segment.end(i))-node.YData(segment.init(i)))^2);
end
% Update length and coordinates on node and segment
segment.len=segment.len.*Len;
root=node.num(node.isroot);
while ~isempty(root)
    rn=[];
    for i=1:length(root)
        if node.isterminal(root(i))
            rn=[rn,[]];
        else
            current=segment.num(segment.init==root(i));
            for k=1:length(current)
                en=segment.end(current(k));
                node.XData(en)=cord(node.XData(root(i)),node.XData(en),Len(current(k)));
                node.YData(en)=cord(node.YData(root(i)),node.YData(en),Len(current(k)));
                rn=[rn,en];
            end
        end
    end
    root=rn;
end
n=length(tta);
% Update theta on the segment
root=node.num(node.isroot);
j=1;
while ~isempty(root)
    rn=[];
    for i=1:length(root)
        if node.isterminal(root(i))
            rn=[rn,[]];
        else
            current=segment.num(segment.init==root(i));
            for k=1:length(current)
                en=segment.end(current(k));
                if segment.isroot(current(k))
                    segment.Gtheta(current(k))=segment.Gtheta(current(k));
                else
                    par=segment.num(segment.end==segment.init(current(k)));
                    segment.Gtheta(current(k))=segment.Gtheta(current(k))+(segment.Gtheta(current(k))-segment.Gtheta(par))*tta(1+mod(j,n))*node.depth(segment.end(current(k)))/max(node.depth);
                end
                node.XData(en) = node.XData(root(i))+distance(current(k))*cos(segment.Gtheta(current(k)));
                node.YData(en) = node.YData(root(i))+distance(current(k))*sin(segment.Gtheta(current(k)));
                j=j+1;
                rn=[rn,en];
            end
        end
    end
    root=rn;
end
% Locate nodes in the capillary bed
X = ceil((node.XData(node.isterminal))./Grid.hx);
Y = ceil(node.YData(node.isterminal)./Grid.hy);

if max(X)>=Grid.Nx | max(Y)>=Grid.Ny | min(X)<=0 | min(Y)<=0
    X(X>Grid.Nx) = Grid.Nx;
    X(X<=0) = 1;
    Y(Y>Grid.Ny) = Grid.Ny;
    Y(Y<=0) = 1;
end

node.input = sub2ind([Grid.Nx Grid.Ny],X,Y);
node.output = node.input;

nod=node;
segmen=segment;

function xen=cord(xin,xen,L)
    xen=xin+(xen-xin)*L;
end
end