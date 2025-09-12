function [Trans,Grid] = FD_Darcy(prm)
A = imread('frog_domain_low.png');
B = im2bw(A);
Bz = size(B);
c = find(B);

Grid = struct('Nx',[]);
Grid.Nx=Bz(1); Grid.hx=1*prm.pixel;
Grid.Ny=Bz(2); Grid.hy=1*prm.pixel;
Grid.Nz=1; Grid.hz=10*prm.pixel;
Grid.K=B.*(prm.perm/prm.mu);
Grid.poro = 0.1*ones(Grid.Nx,Grid.Ny,Grid.Nz);
Grid.volume = Grid.hx*Grid.hy*Grid.hz;
Grid.N = Grid.Nx*Grid.Ny*Grid.Nz;
Grid.B = B;
B = reshape(B,prod(Bz),1);
A = 1-B;

%% Darcy Flow
% Grid sizes
Ny=Grid.Ny; Nx=Grid.Nx; Nz=Grid.Nz; N=Ny*Nx*Nz;
hx=Grid.hx; hy=Grid.hy; hz=Grid.hz; 

L = Grid.K.^(-1);
tx = 2*hy*hz/hx;  ty = 2*hx*hz/hy;

% Transmissibility in x-direction: Nx+1 to give transmissibility at boundaries
Tymat = zeros(Nx,Ny+1); 
Txmat = zeros(Nx+1,Ny);

% Calculate transmissibilities at interfaces (general case- homo, hetero)
Tymat(:,2:Ny) = (ty)./(L(:,1:Ny-1)+L(:,2:Ny)); % Take harmonic mean
Txmat(2:Nx,:) = (tx)./(L(1:Nx-1,:)+L(2:Nx,:)); % Take harmonic mean -- No flow boundary condition: 0 for 1 and Nx+1 -> therefore 2:Nx

% Off-diagonal vectors of the system
% by reshaping transmissibility matrices
Tyvec1 = reshape(Tymat(:,1:Ny), Ny*Nx, 1);
Txvec1 = reshape(Txmat(1:Nx,:), Ny*Nx, 1);
Tyvec2 = reshape(Tymat(:,2:Ny+1), Ny*Nx, 1);
Txvec2 = reshape(Txmat(2:Nx+1,:), Ny*Nx, 1);

% Matrix of the system
Amat = spdiags([-Tyvec1,-Txvec1,Tyvec1+Txvec1+Tyvec2+Txvec2,-Txvec2,-Tyvec2],[Nx,1,0,-1,-Nx],Ny*Nx,Ny*Nx);
Amat = Amat + spdiags(A,0,Ny*Nx,Ny*Nx);
B = spdiags(prm.alpha*B(:),0,Ny*Nx,Ny*Nx);
Am = [Amat+B,-B;-B,prm.beta*Amat+B];
Am(2*Grid.N+1,c(round(Grid.N/10))) = 1;

Trans.ay=Tymat;
Trans.ax=Txmat;
Trans.vy=prm.beta*Tymat;
Trans.vx=prm.beta*Txmat;
Grid.Amat = Am;
end
