function [T,Grid] = GAI_Darcy()
%% Define global variables for optim function
global perm Tymat Txmat Amat cellc Size h 

% Define grid for darcy domain
Grid = struct('Size',[], 'h',[], 'K',[], 'poro', [], 'volume',[] );
Grid.Size =[140 160 1];  % [Nx Ny Nz]
Grid.h=[1 1 1];
Grid.cell.N = prod(Grid.Size);
Grid.cell.num = [1:1:Grid.cell.N]'; 
% Cell center
x = Grid.h(1)/2:Grid.h(1):Grid.h(1)*Grid.Size(1);
y = Grid.h(2)/2:Grid.h(2):Grid.h(2)*Grid.Size(2);
z = Grid.h(3)/2:Grid.h(3):Grid.h(3)*Grid.Size(3);
[xcell, ycell, zcell] = meshgrid(x,y,z);
Grid.cell.cellcenter = [reshape(xcell,Grid.cell.N,1), reshape(ycell,Grid.cell.N,1), reshape(zcell,Grid.cell.N,1)];
Grid.K=ones([3 Grid.Size]);
% define as global variable
Size = Grid.Size;
h = Grid.h;
cellc = Grid.cell.cellcenter;

%% Define permeability
% area=(xcell-).^2+(ycell-ycell(a(i))).^2<=Reps^2;
delta = 10;
BcellInd = find(ycell>=0.5*Size(2)-delta & ycell<=0.5*Size(2)+delta);
% xcell<=0.5*Size(1)+delta & xcell>=0.5*Size(1)-delta
% & ycell>=0.5*Size(2)-delta & ycell<=0.5*Size(2)+delta
Grid.K(:,BcellInd) = 1000; 
perm = Grid.K;

%% Darcy Flow (transmibility)
% Grid sizes
Ny=Grid.Size(2); Nx=Grid.Size(1); N=Grid.cell.N;
hx=Grid.h(1); hy=Grid.h(2); 

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
Tyvec1 = reshape(Tymat(:,1:Ny), Ny*Nx, 1);
Txvec1 = reshape(Txmat(1:Nx,:), Ny*Nx, 1);
Tyvec2 = reshape(Tymat(:,2:Ny+1), Ny*Nx, 1);
Txvec2 = reshape(Txmat(2:Nx+1,:), Ny*Nx, 1);

% Matrix of the system
Amat = spdiags([-Tyvec1,-Txvec1,Tyvec1+Txvec1+Tyvec2+Txvec2,-Txvec2,-Tyvec2],[Nx,1,0,-1,-Nx],Ny*Nx,Ny*Nx);
T.y = Tymat;
T.x = Txmat;
T.Am = Amat;
end