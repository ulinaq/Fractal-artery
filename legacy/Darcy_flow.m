%%% Construction of matrices for solving Darcy as Boundary Value Problem in stagerred grid
%%% Created by Peter Kang @ MIT, modified by Joseph Ma @ NUS, Ulin Nuha @UiB

function [pmat,V] = Darcy_flow(Grid,q)

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

% Off-diagonal vectors of the system
% by reshaping transmissibility matrices
Tyvec1 = reshape(Tymat(:,1:Ny), N, 1);
Txvec1 = reshape(Txmat(1:Nx,:), N, 1);
Tyvec2 = reshape(Tymat(:,2:Ny+1), N, 1);
Txvec2 = reshape(Txmat(2:Nx+1,:), N, 1);

% Matrix of the system
Amat = spdiags([-Tyvec1,-Txvec1,Tyvec1+Txvec1+Tyvec2+Txvec2,-Txvec2,-Tyvec2],[Nx,1,0,-1,-Nx],N,N);
% Amat(1,1) = Amat(1,1)+sum(Grid.K(:,1,1,1));
% inloc = find(q>0);
% a = zeros(1,N);
% a(inloc(1)) = 1;
% Amat = [Amat;a];


% Solve system equation
pvec = Amat\q;
pmat = reshape(pvec,Nx,Ny);

% Fluxes at staggered grid 
V.x = zeros(Nx,Ny+1);
V.y = zeros(Nx+1,Ny);;
V.x(:,2:Ny) = (pmat(:,1:Ny-1)-pmat(:,2:Ny)).*Tymat(:,2:Ny);
V.y(2:Nx,:) = (pmat(1:Nx-1,:)-pmat(2:Nx,:)).*Txmat(2:Nx,:);

V.qx = movmean(V.x,2,2,'Endpoints','discard');
V.qy = movmean(V.y,2,1,'Endpoints','discard');
end
