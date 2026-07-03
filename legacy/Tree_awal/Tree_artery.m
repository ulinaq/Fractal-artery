function [node,segment] = Tree_artery(prm)
%% Create fractal tree
% Create first node (initial node)
node.XData(1) = prm.a_init(1);
node.YData(1) = prm.a_init(2);
node.junction(1) = 9;             % indicate Junction in the node, 9 for root

% Create first segment (first trunk)
segment.init(1) = 1;                 % initial node
segment.end(1) = 2;                  % end node
segment.len(1) = prm.LtoR*prm.Ro;    % length
segment.rad(1) = prm.Ro;             % radius
segment.theta(1) = prm.a_theta;             % angles between each branch of fractal and horizontal axes

% Create second node
node.XData(2) = node.XData(1)+segment.len(1)*cos(segment.theta);
node.YData(2) = node.YData(1)+segment.len(1)*sin(segment.theta);
nodeCount = 2;

% Create remaining segments and nodes
init = 2;
while init <= nodeCount
    kpi = prm.N:-1:1;
%     kpi=randperm(prm.N);
    if segment.rad(init-1)>prm.Rmin
        node.junction(init) = 1;
        for i=1:1:prm.N
            [node,segment,nodeCount] = drawSegment(node,segment,prm,init,nodeCount,i,kpi(i));
        end
    else node.junction(init) = 0;
    end
    init = init+1;
end
node.n = nodeCount;
segment.n = length(segment.len);

% clear gamma i init delta beta kpi; 
function [node,segment,nodeCount] = drawSegment(node,segment,prm,init,nodeCount,i,num)
% segment count always less than node count by 1
bn = sum(node.junction(2:end));
if bn > length(prm.ratio)
    bn = length(prm.ratio);
end
segment.init(nodeCount) = init;
segment.end(nodeCount) = nodeCount+1;
segment.len(nodeCount) = segment.len(init-1)*prm.ratio(bn,num);
segment.rad(nodeCount) = segment.rad(init-1)*prm.ratio(bn,num);
segment.theta(nodeCount) = segment.theta(init-1)+(-1)^i*prm.phi(bn,num);
% create node in the end of new segment
nodeCount = nodeCount+1;
node.Parent(nodeCount) = init;
node.XData(nodeCount) = node.XData(init)+segment.len(nodeCount-1)*cos(segment.theta(nodeCount-1));
node.YData(nodeCount) = node.YData(init)+segment.len(nodeCount-1)*sin(segment.theta(nodeCount-1));
end

end