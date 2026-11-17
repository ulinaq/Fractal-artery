function node=FO_TOFTree(x,node,segment,vnode)
% Compute time of flight on whole nodes in vascular arterial tree
Co = node.n + vnode.n;
Cv = Co + segment.n;

% Update length and coordinates on node and segment
root = node.num(node.isroot);
node.TOF = zeros(node.n,1);
while ~isempty(root)
    rn=[];
    for i=1:length(root)
        if node.isterminal(root(i))
            rn=[rn,[]];
        else
            current=segment.num(segment.init==root(i));
            for k=1:length(current)
                en=segment.end(current(k));
                node.TOF(en)=node.TOF(root(i))+segment.len(current(k))*pi*segment.rad(current(k))^2/x(Co+current(k));
                rn=[rn,en];
            end
        end
    end
    root=rn;
end
end
