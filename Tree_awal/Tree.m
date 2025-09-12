function [node,vnode,segment,vsegment] = Tree(prm)

Dz = size(prm.Domain);
De = prod(Dz);

node.avc = [50,50,20;45,45,15]; 
node.XData = node.avc.*prm.pixel; 
node.isterminal = [0,1]; 
node.junction = [0,0]; 
node.Parent = [0,0]; 
node.n = 2; 
node.isroot = [1,0]; 
node.num = 1:node.n; 
segment.init = 1;
segment.end = 2;
segment.len = 4*sqrt(3);
segment.rad = 5;
segment.n =1;
segment.num =1;
segment.isroot = 1;
segment.isterminal = 1;


vnode.avc = [1,1,1;5,5,5];
vnode.XData = vnode.avc.*prm.pixel; 
vnode.isterminal = [0,1]; 
vnode.junction = [0,0]; 
vnode.Parent = [0,0]; 
vnode.n = 2; 
vnode.isroot = [1,0]; 
vnode.num = 1:vnode.n; 
vsegment.init = 1;
vsegment.end = 2;
vsegment.len = 4*sqrt(3);
vsegment.rad = 8;
vsegment.n =1;
vsegment.num =1;
vsegment.isroot = 1;
vsegment.isterminal = 1;

node.input = sub2ind(Dz,node.avc(2,1),node.avc(2,2),node.avc(2,3));
vnode.output = sub2ind(Dz,vnode.avc(2,1),vnode.avc(2,2),vnode.avc(2,3));
end