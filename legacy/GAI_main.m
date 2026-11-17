%% Domain Parameters
[T,Grid] = GAI_Darcy();
% Tree
alpha = [0.8,0.7];
prm = GAI_parameter(alpha);    % Set initial value and tree parameter

[node,vnode,segment,vsegment] = tree_init(prm);

%% Main iteration (create branches)
% Create junction on the outer node, compare each junction to be optimized.
% Decide 1 of several node to be a new junction,
[node,vnode,segment,vsegment] = tree_junction(prm,node,vnode,segment,vsegment);
[node,vnode,segment,vsegment] = tree_branch(prm,node,vnode,segment,vsegment,1);
function [node,vnode,segment,vsegment] = tree_junction(prm,node,vnode,segment,vsegment)
% Find end node
in = node.junction==0;
out = vnode.junction==0;
temp = node;
% Check the radii size
% Create a temporary branch (left as default) 
% Check coordinate of branch, if out of boundary note it 
% Compute ToF --> need function for computing ToF at certain tree
% ToF(prm,Grid,node,vnode)
% repeat for temporary branch (right), and compare both solution and
% compare the result --> choose better branch as real branches
% 

node = temp;
end


%% Make branch
% make branches in artery, consider TOF to decide which side the longest
% branch. Create both option branches and compare the result, choose the
% optimal one for the branches --> save. 
function [node,vnode,segment,vsegment] = tree_branch(prm,node,vnode,segment,vsegment,kiri)
% logical kiri for deciding longest branch in bifurcation
if kiri==1
    kpi = prm.N:-1:1;
else
    kpi = 1:1:prm.N;
end
init=2; % belum ..............
node.junction(init) = 1;
for i=1:1:prm.N
   [node,segment] = drawSegment(node,segment,prm,init,i,kpi(i));
end

end
% limiting factor for junction are radii and flow rate and of course
% capillary domain in the simulation.


function [node,vnode,segment,vsegment] = tree_init(prm)
%% artery
% Create first node (initial node)
node.XData(1) = prm.a_init(1);
node.YData(1) = prm.a_init(2);
node.junction(1) = 9;             % indicate Junction in the node, 9 for root 

% Create first segment (first trunk)
segment.init(1) = 1;                 % initial node
segment.end(1) = 2;                  % end node
segment.len(1) = prm.LtoR*prm.Ro;    % length
segment.rad(1) = prm.Ro;             % radius
segment.theta(1) = prm.a_theta;      % angles between each branch of fractal and horizontal axes

% Create second node
node.XData(2) = node.XData(1)+segment.len(1)*cos(segment.theta);
node.YData(2) = node.YData(1)+segment.len(1)*sin(segment.theta);
node.n = 2;
%% Vein
% Create first vnode (initial vnode)
vnode.XData(1) = prm.v_init(1);
vnode.YData(1) = prm.v_init(2);
vnode.junction(1) = 9;             % indicate Junction in the vnode, 9 for root

% Create first vsegment (first trunk)
vsegment.init(1) = 1;                 % initial vnode
vsegment.end(1) = 2;                  % end vnode
vsegment.len(1) = prm.LtoR*prm.Rvo;    % length
vsegment.rad(1) = prm.Rvo;             % radius
vsegment.theta(1) = prm.v_theta;       % angles between each branch of fractal and vertical axes

% Create second vnode
vnode.XData(2) = vnode.XData(1)+vsegment.len*cos(vsegment.theta);
vnode.YData(2) = vnode.YData(1)+vsegment.len*sin(vsegment.theta);
vnode.n = 2;
end

% node.n = the last node in the tree
function [node,segment] = drawSegment(node,segment,prm,init,i,num)
% segment n always less than node n by 1
segment.init(node.n) = init;
segment.end(node.n) = node.n+1;
segment.len(node.n) = segment.len(init-1)*prm.ratio(num);
segment.rad(node.n) = segment.rad(init-1)*prm.ratio(num);
segment.theta(node.n) = segment.theta(init-1)+(-1)^i*prm.phi(num);
% create node in the end of new segment
node.n = node.n+1;
node.Parent(node.n) = init;
node.XData(node.n) = node.XData(init)+segment.len(node.n-1)*cos(segment.theta(node.n-1));
node.YData(node.n) = node.YData(init)+segment.len(node.n-1)*sin(segment.theta(node.n-1));
end
