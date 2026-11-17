function [P,pvec,T] = Darcy_flow_superposition(Grid,input,output,Pin,Reps)
% clear all;
% Grid.Nx=40; Grid.hx=1/Grid.Nx;
% Grid.Ny=40; Grid.hy=1/Grid.Ny;
% Grid.Nz=1; Grid.hz=1/Grid.Nz;
% Grid.K=exp(5*smooth3(smooth3(randn(3,Grid.Nx,Grid.Ny))));
% Grid.K=ones(3,Grid.Nx,Grid.Ny);
% N=Grid.Nx*Grid.Ny*Grid.Nz;

% Grid sizes
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
hx=Grid.hx; hy=Grid.hy; 

L = Grid.K.^(-1);
tx = 2*hy/hx;  ty = 2*hx/hy;

% Cell center
x = hx/2:hx:hx*Nx;
y = hy/2:hy:hy*Ny;
[ycell, xcell]=meshgrid(y,x);

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
% Amat(1,1) = Amat(1,1)+sum(Grid.K(:,1,1,1));
% a = round([0.2 0.2 0.8 0.8]*N + [0.3*Ny 0.7*Ny -0.3*Ny -0.7*Ny]);
% b = [0.2; 0.8; -0.4; -0.6]*1e-2;
% q=zeros(N+1,1);   % without source as default
% for i=1:length(a)
% q(a(i))=b(i);
% end
Amat(N+1,1) = 1;
% q(N+1) = 0;
% pvec = Amat\q;
% pmat = reshape(pvec,Nx,Ny);
a = [input,output];
z = zeros(Nx,Ny);
for i=1:length(a)
    % Create circle region for spreading epsilon
    z=Reps+1-(xcell-xcell(a(i))).^2-(ycell-ycell(a(i))).^2;
    area=(xcell-xcell(a(i))).^2+(ycell-ycell(a(i))).^2<=Reps;
    zn=z(area)*hx*hy/sum(z(area)*hx*hy);
q=zeros(N+1,1);   % without source as default
q(area)=zn;
q(N+1) = 0;       % dirichlet BC set 0 at corner (0,0)
% Solve system equation
pvec(:,i) = Amat\q;
end

% P=[pvec(input,:); -1*pvec(output,:)];
P=pvec([input,output],:);
% P = pvec*b;
% 
% Pmat = reshape(P,Nx,Ny);
% Perr = pmat-Pmat;
% Pn = norm(Perr);
T.y=Tymat;
T.x=Txmat;
end
% nn=0:Ny-1;
% pex=1-nn./(Ny-1);
% error=pmat(1:Nx:N)-pex;
% figure()
% subplot(1,2,1)
% contourf(pmat,20); colorbar; title('Pressure');
% subplot(1,2,2)
% contourf(Pmat,20); colorbar; title('Pressure');

