% clear all;
tic;

%% Parameters
prm = Tree_parameter();

%% Create Fractal tree
% Arterial tree
[node,segment] = Tree_artery(prm);
% venous tree
[vnode,vsegment] = Tree_vein(prm);

% Make visualization
GV=graph(vsegment.init,vsegment.end);
GA=graph(segment.init,segment.end);

figure()
% subplot(1,2,1)
plot(GA,'XData',node.XData,'YData',node.YData);
title('Artery');
daspect([1 1 1])
hold on;
% subplot(1,2,2)
plot(GV,'XData',vnode.XData,'YData',vnode.YData);
title('Vein');
daspect([1 1 1])

%% Capillary Domain
[P,T,pvec,Grid,node,vnode] = Capillary(node, segment, vnode, vsegment, prm,Pin);

%% Calculate pressure in nodes (Update April 26, 2018)
[node,segment,x] = simpleEquation(node,prm,segment,vnode,vsegment,Pin,Qin,P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:node.n+vnode.n;
q = [x([Co+seg(node.in)-1]);-1*x([Cv+seg(vnode.out)-1])];
Pmat = x(Co)+pvec*q;
% ppp(end+1,:)=Pmat;
% end
Pmat = reshape(Pmat,Grid.Nx,Grid.Ny);
V = Flux(Pmat,Grid,T)

% Use linear system (Poiseuille) for initial value (Update March 2018)
% Xlin = LinTreeSolver(prm,node,segment,Pin,Qin,nodeCount);
% [node,segment,Xnonlin] = TreeSysEquation(Xlin,node,prm,segment,nodeCount,Pin,Qin);
% [vnode,vsegment,x] = VenousSysEquation(vnode,prm,vsegment,vnodeCount,Pin,Qin);
% ratio = Calrat(node,segment,nodeCount);

figure()
contourf(Pmat,20); colorbar; title('Pressure');
hold on;
quiver(V.qx,V.qy,'r'); %title('Fluxes');
toc
% [node,segment] = Pressure(node,prm,segment,nodeCount);

%% Capillary domain (Darcy)
% Create domain
function [P,T,pvec,Grid,node,vnode] = Capillary(node, segment, vnode, vsegment, prm,Pin)
in = node.junction==0;
in(1) = 0;
node.in = in;
out = vnode.junction==0;
out(1) = 0;
vnode.out = out;
DomainSize = [120 160]; % x and y
DomainLimit = [-60 60 0 160]; % xmin, xmax, ymin ymax
Reps = 3;   % Epsilon radii for dirac

% Grid for Darcy domain
Grid.Nx=120; Grid.hx=DomainSize(1)/Grid.Nx;
Grid.Ny=160; Grid.hy=DomainSize(2)/Grid.Ny;
Grid.Nz=1; Grid.hz=1;
Grid.K=ones(3,Grid.Nx,Grid.Ny);

X = ceil((node.XData(in)- DomainLimit(1))./Grid.hx);
Y = ceil((node.YData(in)- DomainLimit(3))./Grid.hy);
Xout = floor((vnode.XData(out)-DomainLimit(1))./Grid.hx);
Yout = floor((vnode.YData(out))./Grid.hy);

input = sub2ind([Grid.Nx Grid.Ny],X,Y);
output = sub2ind([Grid.Nx Grid.Ny],Xout,Yout);

[P,pvec,T] = Darcy_flow_superposition(Grid,input,output,Pin,Reps);
% [P,V]=Darcy_flow(Grid,q);
end

function [segment,node] = ChangeTheta(segment,node,thetaChange)
for i = 2:node.n
    % Change theta
    phi=segment.theta(segment.init==i)-segment.theta(segment.end==i);
    phi=thetaChange*phi;
    theta(segment.init==i)=segment.theta(segment.end==i)+phi;
    % Redrawing segment
    ch = segment.end(segment.init==i);
    node.XData(ch) = node.XData(i)+segment.len(segment.init==i).*cos(theta(segment.init==i));
    node.YData(ch) = node.YData(i)+segment.len(segment.init==i).*sin(theta(segment.init==i));
end
segment.theta=theta;
end

   function V = Flux(Pmat,Grid,T)
       % Fluxes at staggered grid
    Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
        
    V.x = zeros(Nx,Ny+1);
    V.y = zeros(Nx+1,Ny);;
    V.x(:,2:Ny) = (Pmat(:,1:Ny-1)-Pmat(:,2:Ny)).*T.y(:,2:Ny);
    V.y(2:Nx,:) = (Pmat(1:Nx-1,:)-Pmat(2:Nx,:)).*T.x(2:Nx,:);

    V.qx = movmean(V.x,2,2,'Endpoints','discard');
    V.qy = movmean(V.y,2,1,'Endpoints','discard');
    V.u = V.qx.^2+V.qy.^2;
    u = reshape(V.u,N,1);
    V.range = max(u)-min(u);
   end