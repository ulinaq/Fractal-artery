function [vnode,vsegment] = Frog_vein(prm)
%% Create fractal tree
% Create first vnode (initial vnode)
vnode.XData(1) = prm.v_init(1);
vnode.YData(1) = prm.v_init(2);
vnode.junction(1) = 0;             % indicate Junction in the vnode, 9 for root
vnode.isroot(1) = 1;
vnode.isterminal(1) = 0;
% Create first vsegment (first trunk)
vsegment.init(1) = 1;                 % initial vnode
vsegment.end(1) = 2;                  % end vnode
vsegment.len(1) = prm.LtoR*prm.Rvo;    % length
vsegment.rad(1) = prm.Rvo;             % radius
vsegment.theta(1) = prm.v_theta;       % angles between each branch of fractal and vertical axes

% Create second vnode
vnode.XData(2) = vnode.XData(1)+vsegment.len*cos(vsegment.theta);
vnode.YData(2) = vnode.YData(1)+vsegment.len*sin(vsegment.theta);
vnode.Parent(2) = 1;
vnodeCount = 2;

% Create remaining vsegments and vnodes
init = 2;
target =[140,80];

while init <= vnodeCount
x1=[vnode.XData(vnode.Parent(init)), vnode.YData(vnode.Parent(init))];
x2=[vnode.XData(init), vnode.YData(init)];
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
    if vsegment.rad(init-1)>prm.Rvmin
        vnode.junction(init) = 1;
        vnode.isterminal(init) = 0;
        for i=1:1:prm.N
            [vnode,vsegment,vnodeCount] = drawvsegment(vnode,vsegment,prm,init,vnodeCount,i,kpi(i));
        end
    else
        vnode.junction(init) = 0;
        vnode.isterminal(init) = 1;
    end
    init = init+1;
end
vnode.n = vnodeCount;
vsegment.n = length(vsegment.len);
vnode.isroot(2:vnodeCount) = 0;
vnode.isroot = logical(vnode.isroot);
vnode.isterminal = logical(vnode.isterminal);
clear gamma i init delta beta kpi; 

function [vnode,vsegment,vnodeCount] = drawvsegment(vnode,vsegment,prm,init,vnodeCount,i,num)
% vsegment count always less than vnode count by 1
vsegment.init(vnodeCount) = init;
vsegment.end(vnodeCount) = vnodeCount+1;
vsegment.len(vnodeCount) = vsegment.len(init-1)*prm.vratio(num);
vsegment.rad(vnodeCount) = vsegment.rad(init-1)*prm.vratio(num);
vsegment.theta(vnodeCount) = vsegment.theta(init-1)+(-1)^i*prm.vphi(num);
% create vnode in the end of new vsegment
vnodeCount = vnodeCount+1;
vnode.Parent(vnodeCount) = init;
vnode.XData(vnodeCount) = vnode.XData(init)+vsegment.len(vnodeCount-1)*cos(vsegment.theta(vnodeCount-1));
vnode.YData(vnodeCount) = vnode.YData(init)+vsegment.len(vnodeCount-1)*sin(vsegment.theta(vnodeCount-1));
end

end