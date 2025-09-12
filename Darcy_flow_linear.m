% function [pmat,V,q] = Darcy_flow2(Grid,q,Pin)
% clear all;
    Grid.Nx=40; Grid.hx=1/Grid.Nx;
    Grid.Ny=40; Grid.hy=1/Grid.Ny;
    Grid.Nz=1; Grid.hz=1/Grid.Nz;
%     Grid.K=exp(5*smooth3(smooth3(randn(3,Grid.Nx,Grid.Ny))));
    Grid.K=ones(3,Grid.Nx,Grid.Ny);
    N=Grid.Nx*Grid.Ny*Grid.Nz;

% Grid sizes
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
hx=Grid.hx; hy=Grid.hy; 

L = Grid.K.^(-1);
tx = 2*hy/hx;  ty = 2*hx/hy;

% Transmissibility in x-direction: Nx+1 to give transmissibility at boundaries
Tymat = zeros(Nx,Ny+1); 
Txmat = zeros(Nx+1,Ny);

% Calculate transmissibilities at interfaces (general case- homo, hetero)
Tymat(:,2:Ny) = (ty)./(L(1,:,1:Ny-1)+L(1,:,2:Ny)); % Take harmonic mean
Txmat(2:Nx,:) = (tx)./(L(2,1:Nx-1,:)+L(2,2:Nx,:)); % Take harmonic mean -- No flow boundary condition: 0 for 1 and Nx+1 -> therefore 2:Nx

% % No boundary on x=0 and x=Nx
% Txmat(1,:) = (tx)./(L(2,1,:)+L(2,Nx,:)); % Take harmonic mean -- No flow boundary condition: 0 for 1 and Nx+1 -> therefore 2:Nx
% Txmat(Nx+1,:) = Txmat(1,:);
% No boundary on y=0 and y=Ny
% Tymat(:,1) = (ty)./(L(1,:,1)+L(1,:,Ny)); % Take harmonic mean -- No flow boundary condition: 0 for 1 and Nx+1 -> therefore 2:Nx
% Tymat(:,Ny+1) = Tymat(:,1);

% Off-diagonal vectors of the system
% by reshaping transmissibility matrices
Tyvec1 = reshape(Tymat(:,1:Ny), Ny*Nx, 1);
Txvec1 = reshape(Txmat(1:Nx,:), Ny*Nx, 1);
Tyvec2 = reshape(Tymat(:,2:Ny+1), Ny*Nx, 1);
Txvec2 = reshape(Txmat(2:Nx+1,:), Ny*Nx, 1);

% Matrix of the system
Amat = spdiags([-Tyvec1,-Txvec1,Tyvec1+Txvec1+Tyvec2+Txvec2,-Txvec2,-Tyvec2],[Nx,1,0,-1,-Nx],Ny*Nx,Ny*Nx);
% Amat(1,1) = Amat(1,1)+sum(Grid.K(:,1,1,1));
q=zeros(N,1);   % without source as default

% Arterial tree system equation (4 nodes and 3 trunks= 5 unknowns and 2 known,
% Pin = 1 and Qin = 1), 2 Q and 3 P are unknowns 
Pin = 1; Qin = 1;
Len = [0.25,0.175,0.209];
Rad = [2,1.4,1.67];
Res = 8*Len./(Rad.^4);
Art = zeros(4,5);
b = zeros(4,1);
b(1) = Pin - Res(1)*Qin;
b(2) = Qin;
Art([1 3 4],3) = 1;
Art(2,[1 2]) = 1;
Art(3,4) = -1; Art(3,1) = -Res(2);
Art(4,5) = -1; Art(4,2) = -Res(3);

% Define arterial point sources on darcy domain 
inD = ceil(N/4-Ny*[-2/3 5/3]);
ArD = zeros(N,5);
ArD(inD,1) = -1; 
ArZ = zeros(4,N);

% Combine
A = [Amat, ArD; ArZ, Art];

% Pressure relation
A(N+5,inD(1)) = 1; A(N+5,N+4) = -1; 
A(N+6,inD(2)) = 1; A(N+6,N+5) = -1; 
q = [q;b;zeros(2,1)];

% Venous tree system equation (4 nodes and 3 trunks= 6 unknowns and 1 known,
% Qout = Qin), 2 Q and 4 P are unknowns 
Qout = Qin;
LenV = [0.375,0.2625,0.3135];
RadV = [3,2.1,2.505];
ResV = 8*LenV./(RadV.^4);
Ve = zeros(4,6);
b = zeros(4,1);
b(1) = ResV(1)*Qout;
b(2) = Qout;
Ve(1,3) = -1; Ve([1 3 4],4) = 1;
Ve(2,[1 2]) = 1;
Ve(3,5) = -1; Ve(3,1) = ResV(2);
Ve(4,6) = -1; Ve(4,2) = ResV(3);

% Define Vein point sinks on darcy domain 
VeD = zeros(N+6,6);
outV = ceil(3*N/4+Ny*[-2/7 7/5]);
VeD(outV,1) = 1; 
VeZ = zeros(4,N+5);

% Combine
A = [A, VeD; VeZ, Ve];

% Pressure relation
A(N+11,outV(1)) = 1; A(N+11,N+10) = -1; 
A(N+12,outV(2)) = 1; A(N+12,N+11) = -1; 
q = [q;b;zeros(2,1)];

% Solve system equation
pvec = A\q;
pmat = reshape(pvec(1:N),Nx,Ny);
Pout = pvec(N+8);

% Fluxes at staggered grid 
V.x = zeros(Nx,Ny+1);
V.y = zeros(Nx+1,Ny);;
V.x(:,2:Ny) = (pmat(:,1:Ny-1)-pmat(:,2:Ny)).*Tymat(:,2:Ny);
V.y(2:Nx,:) = (pmat(1:Nx-1,:)-pmat(2:Nx,:)).*Txmat(2:Nx,:);

V.qx = movmean(V.x,2,2,'Endpoints','discard');
V.qy = movmean(V.y,2,1,'Endpoints','discard');
% end
nn=0:Ny-1;
pex=1-nn./(Ny-1);
error=pmat(1:Nx:N)-pex;
% figure()
% contourf(pmat,20); colorbar; title('Pressure');
figure()
contourf(pmat,20); colorbar; title('Pressure');
hold on;
quiver(V.qx,V.qy,'r');
% subplot(1,2,2)
% plot(1:Ny,pmat(1:Nx:N),1:Ny,pex);
% figure()
% quiver(V.qx,V.qy);