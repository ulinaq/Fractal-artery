function [node,segment] = Frog_artery(prm)
%% Create fractal tree
% Create first node (initial node)
node.XData(1) = prm.a_init(1);
node.YData(1) = prm.a_init(2);
node.junction(1) = 0;             % indicate Junction in the node, 9 for root
node.isroot(1) = 1;
node.isterminal(1) = 0;
% Create first segment (first trunk)
segment.init(1) = 1;                 % initial node
segment.end(1) = 2;                  % end node
segment.len(1) = prm.LtoR*prm.Ro;    % length
segment.rad(1) = prm.Ro;             % radius
segment.theta(1) = prm.a_theta;             % angles between each branch of fractal and horizontal axes

% Create second node
node.XData(2) = node.XData(1)+segment.len(1)*cos(segment.theta);
node.YData(2) = node.YData(1)+segment.len(1)*sin(segment.theta);
node.Parent(2) = 1;
nodeCount = 2;

% Create remaining segments and nodes
target =[0,80];
init = 2;
while init <= nodeCount
x1=[node.XData(node.Parent(init)), node.YData(node.Parent(init))];
x2=[node.XData(init), node.YData(init)];
len1=x2-x1;
len2=target-x2;
theta1=atan(len1(2)/len1(1));
theta2=atan(len2(2)/len2(1));
if theta2>theta1
    kiri=1;
else
    kiri=0;
end
if kiri==1
    kpi = prm.N:-1:1;
else
    kpi = 1:1:prm.N;
end
%     kpi=randperm(prm.N);
    if segment.rad(init-1)>prm.Rmin
        node.junction(init) = 1;
        node.isterminal(init) = 0;
        for i=1:1:prm.N
            [node,segment,nodeCount] = drawSegment(node,segment,prm,init,nodeCount,i,kpi(i));
        end
    else
        node.isterminal(init) = 1;
        node.junction(init) = 0;
    end
    init = init+1;
end
node.n = nodeCount;
segment.n = length(segment.len);
node.isroot(2:nodeCount) = 0;
node.isroot = logical(node.isroot);
node.isterminal = logical(node.isterminal);

% clear gamma i init delta beta kpi; 
function [node,segment,nodeCount] = drawSegment(node,segment,prm,init,nodeCount,i,num)
% segment count always less than node count by 1
segment.init(nodeCount) = init;
segment.end(nodeCount) = nodeCount+1;
segment.len(nodeCount) = segment.len(init-1)*prm.ratio(num);
segment.rad(nodeCount) = segment.rad(init-1)*prm.ratio(num);
segment.theta(nodeCount) = segment.theta(init-1)+(-1)^i*prm.phi(num);
% create node in the end of new segment
nodeCount = nodeCount+1;
node.Parent(nodeCount) = init;
node.XData(nodeCount) = node.XData(init)+segment.len(nodeCount-1)*cos(segment.theta(nodeCount-1));
node.YData(nodeCount) = node.YData(init)+segment.len(nodeCount-1)*sin(segment.theta(nodeCount-1));
end

end