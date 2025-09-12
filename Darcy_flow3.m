% Darcy with boundary on ghost cell
% function [pmat,V,q] = Darcy_flow2(Grid,q,Pin)
clear all;
    Grid.Nx=20; Grid.hx=1/Grid.Nx;
    Grid.Ny=20; Grid.hy=1/Grid.Ny;
    Grid.Nz=1; Grid.hz=1/Grid.Nz;
    Grid.K=ones(3,Grid.Nx,Grid.Ny);
    N=Grid.Nx*Grid.Ny*Grid.Nz;
% Grid sizes
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
hx=Grid.hx; hy=Grid.hy; 

% cell center
x = hx/2:hx:1;
y = hy/2:hy:1;
[ycell, xcell]=meshgrid(x,y);

L = Grid.K.^(-1);
tx = 2*hy/hx;  ty = 2*hx/hy;

% Transmissibility in x-direction: Nx+1 to give transmissibility at boundaries
Tymat = zeros(Nx,Ny+1); 
Txmat = zeros(Nx+1,Ny);

% Calculate transmissibilities at interfaces (general case- homo, hetero)
Tymat(:,2:Ny) = (ty)./(L(1,:,1:Ny-1)+L(1,:,2:Ny)); % Take harmonic mean
Txmat(2:Nx,:) = (tx)./(L(2,1:Nx-1,:)+L(2,2:Nx,:)); % Take harmonic mean -- No flow boundary condition: 0 for 1 and Nx+1 -> therefore 2:Nx

% Transmibility Dirichlet BC left and right ghost cell
Txmat(1,:) = (tx)./L(2,1,:);
Txmat(Nx+1,:) = (tx)./L(2,end,:);
Txmat = [zeros(1,Ny);Txmat;zeros(1,Ny)];
% Tymat = [zeros(1,Ny+1);Tymat;zeros(1,Ny+1)];
Tymat = [Tymat(1,:);Tymat;Tymat(Nx,:)];
% Transmibility Dirichlet BC left and right ghost cell
Tymat(:,1) = 2*Tymat(:,2);
Tymat(:,Ny+1) = 2*Tymat(:,Ny);
Tymat = [zeros(Nx+2,1),Tymat,zeros(Nx+2,1)];
% Tymat = [zeros(1,Ny+1);Tymat;zeros(1,Ny+1)];
Txmat = [Txmat(:,1),Txmat,Txmat(:,Nx)];
N2=(Ny+2)*(Nx+2);   % Dirichlet BC on x and y
% N2=Ny*(Nx+2);     % Dirichlet BC on x

% Off-diagonal vectors of the system
% by reshaping transmissibility matrices
Tyvec1 = reshape(Tymat(:,1:end-1), N2, 1);
Txvec1 = reshape(Txmat(1:end-1,:), N2, 1);
Tyvec2 = reshape(Tymat(:,2:end), N2, 1);
Txvec2 = reshape(Txmat(2:end,:), N2, 1);

% Matrix of the system
Amat = spdiags([-Tyvec1,-Txvec1,Tyvec1+Txvec1+Tyvec2+Txvec2,-Txvec2,-Tyvec2],[Nx+2,1,0,-1,-Nx-2],N2,N2);

% add source
% src = [2*xcell.*(1-xcell)+2*ycell.*(1-ycell)]; 
src = [pi^2*sin(pi*xcell).*cos(pi*ycell)+pi^2*sin(pi*xcell).*cos(pi*ycell)];

q = [0,zeros(1,Nx),0;zeros(Ny,1),src,zeros(Ny,1);0,zeros(1,Nx),0]*hx*hy;
Q = reshape(q,N2,1);
% Set Dirichlet BC at P(x=0)=0 and P(x=1)=0
for i=1:Nx+2:N2
% q(i) = 1;
Amat(i,:) = 0;
Amat(i,i) = 1;
end
for i=Nx+2:Nx+2:N2
Amat(i,:) = 0;
Amat(i,i) = 1;
end
% Set Dirichlet BC at P(y=0)=0 and P(y=1)=0
for i=1:Ny+2 % y=0
if i>1 & i<Ny+2
    Q(i) = sin(pi*xcell(i-1,1));
end
Amat(i,:) = 0;
Amat(i,i) = 1;
end
for i=N2-Ny-1:N2 % y=1
    if i>N2-Ny-1 & i<N2
    Q(i) = -sin(pi*xcell(i-N2+Ny+1));
    end
Amat(i,:) = 0;
Amat(i,i) = 1;
end

% Solve system equation
pvec = Amat\Q;
pmat = reshape(pvec,Nx+2,Ny+2);
P = pmat(2:Nx+1,2:Ny+1);
 
% nn=hy/2:hy:1;
% pex=1-nn.^2;
% pex = [xcell.*(1-xcell).*ycell.*(1-ycell)];
pex = sin(pi*xcell).*cos(pi*ycell);
error=P-pex;
norm(error)
figure()
subplot(1,2,1)
contourf(P,20); colorbar; title('Pressure');
subplot(1,2,2)
contourf(pex,20); colorbar; title('Pressure Exact');
% plot(nn,P(1:Ny),nn,pex);