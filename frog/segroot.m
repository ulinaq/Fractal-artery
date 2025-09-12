function segment = segroot(node,segment)
seg = 1:2*node.n;
root = find(node.isroot);
rot=[];
for k=1:length(root)
    rot=[rot,seg(segment.init==root(k))];
end
segment.isroot = logical(zeros(segment.n,1));
segment.isroot(rot) = 1;
root = find(node.isterminal);
rot=[];
for k=1:length(root)
    rot=[rot,seg(segment.end==root(k))];
end
segment.isterminal = logical(zeros(segment.n,1));
segment.isterminal(rot) = 1;
end
