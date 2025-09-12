function MinF = OptimTree(alpha)
%% Parameters
prm = Tree_parameter(alpha);
% prm.phi(1:2)=alpha(1)/2;
% prm.vphi(1:2)=alpha(2)/2;

%% Create Fractal tree
% Arterial tree
[node,segment] = Tree_artery(prm);
% venous tree
[vnode,vsegment] = Tree_vein(prm);

%% Capillary Domain
% domain size 
[Xr,T,Grid,node,vnode] = Tree_Capillary_Darcy(node, vnode);
if isnan(Xr.P)
    MinF=inf;
    return;
end

%% Calculate pressure in nodes (Update April 26, 2018)
x = Tree_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% [node,segment,x] = Tree_nonlinEq(node,prm,segment,vnode,vsegment,P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:Co;
q = [x([Co+seg(node.in)-1]);-1*x([Cv+seg(vnode.out)-1])];
Pmat = x(Co)+Xr.pvec*q;
Pmat = reshape(Pmat,Grid.Nx,Grid.Ny);
V = Flux(Pmat,Grid,T);
MinF = x(node.n);

function V = Flux(Pmat,Grid,T)
       % Fluxes at staggered grid
    Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
        
    V.x = zeros(Nx,Ny+1);
    V.y = zeros(Nx+1,Ny);
    V.x(:,2:Ny) = (Pmat(:,1:Ny-1)-Pmat(:,2:Ny)).*T.y(:,2:Ny);
    V.y(2:Nx,:) = (Pmat(1:Nx-1,:)-Pmat(2:Nx,:)).*T.x(2:Nx,:);

    V.qx = movmean(V.x,2,2,'Endpoints','discard');
    V.qy = movmean(V.y,2,1,'Endpoints','discard');
    V.u = sqrt(V.qx.^2+V.qy.^2);
    u = reshape(V.u,N,1);
    V.range = max(u)-min(u);
    V.minF = norm((V.u-mean(u)),'fro');
    V.std = std(u);
    V.mean = mean(u);
   end

end