function [Trans,Grid,state] = FO_Darcy(prm,node,vnode)
Bz = size(prm.Domain);
B = prm.Domain;

Grid = struct('Nx',[]);
Grid.Nx=Bz(1); Grid.hx=1*prm.pixel;
Grid.Ny=Bz(2); Grid.hy=1*prm.pixel;
Grid.Nz=Bz(3); Grid.hz=1*prm.pixel;
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
tx = 2*hy*hz/hx;  ty = 2*hx*hz/hy;  ty = 2*hx*hy/hz;

% Transmissibility in x-direction: Nx+1 to give transmissibility at boundaries
Txmat = zeros(Nx+1,Ny,Nz);
Tymat = zeros(Nx,Ny+1,Nz); 
Tzmat = zeros(Nx,Ny,Nz+1);

% Calculate transmissibilities at interfaces (general case- homo, hetero)
Txmat(2:Nx,:,:) = (tx)./(L(1:Nx-1,:,:)+L(2:Nx,:,:)); % Take harmonic mean -- No flow boundary condition: 0 for 1 and Nx+1 -> therefore 2:Nx
Tymat(:,2:Ny,:) = (ty)./(L(:,1:Ny-1,:)+L(:,2:Ny,:)); % Take harmonic mean
Tzmat(:,:,2:Nz) = (ty)./(L(:,:,1:Nz-1)+L(:,:,2:Nz)); % Take harmonic mean

% Off-diagonal vectors of the system
% by reshaping transmissibility matrices
Txvec1 = reshape(Txmat(1:Nx,:,:), N, 1);
Tyvec1 = reshape(Tymat(:,1:Ny,:), N, 1);
Tzvec1 = reshape(Tzmat(:,:,1:Nz), N, 1);
Txvec2 = reshape(Txmat(2:Nx+1,:,:), N, 1);
Tyvec2 = reshape(Tymat(:,2:Ny+1,:), N, 1);
Tzvec2 = reshape(Tzmat(:,:,2:Nz+1), N, 1);

% Matrix of the system
Amat = spdiags([-Tzvec1,-Tyvec1,-Txvec1,Tzvec1+Tyvec1+Txvec1+Txvec2+Tyvec2+Tzvec2,-Txvec2,-Tyvec2,-Tzvec2],[Nx*Ny,Nx,1,0,-1,-Nx,-Nx*Ny],N,N);
Amat = Amat + spdiags(A,0,N,N);
B = spdiags(prm.alpha*B(:),0,N,N);
Am = [Amat+B,-B;-B,prm.beta*Amat+B];
Am(2*Grid.N+1,1) = 1;

Trans.ax=Txmat;
Trans.ay=Tymat;
Trans.az=Tzmat;
Trans.vx=prm.beta*Txmat;
Trans.vy=prm.beta*Tymat;
Trans.vz=prm.beta*Tzmat;
% Grid.Amat = Am;

%% Capillary
% Cell center
x = hx/2:hx:hx*Nx;
y = hy/2:hy:hy*Ny;
z = hz/2:hz:hz*Nz;
[xcell, ycell, zcell]=meshgrid(x,y,z);
Reps = 10*prm.pixel;
a = [node.input;vnode.output];
zn = [];
Grid.src.cell = [];
Grid.src.rate = [];
Grid.cell = [1:N]';
% Create Block A_T-X
% ATX=zeros(length(a),2*Grid.N);   % ATX = AXT'
% Create Block A_T-X
AX=zeros(Grid.N,length(node.input));   % without source as default
AT=zeros(Grid.N,length(vnode.output));   % without source as default
j=1;
for i=1:length(a)
    % Create circle region for spreading epsilon
    area=(xcell-xcell(a(i))).^2+(ycell-ycell(a(i))).^2+(zcell-zcell(a(i))).^2<=Reps^2;
    zn=exp(-1./(1-((xcell(area)-xcell(a(i))).^2+(ycell(area)-ycell(a(i))).^2+(zcell(area)-zcell(a(i))).^2)));
    Grid.src.cell{i} = Grid.cell(area);
    zn=zn/sum(zn);
    if i <= length(node.input)
        AX(area,i)=zn;
    else
        AT(area,j)=zn;
        j=j+1;
    end
    Grid.src.rate{i} = zn;
end
AXT = [AX,zeros(Grid.N,length(vnode.output));zeros(Grid.N,length(node.input)),AT];
state.ATX = sparse([1*AXT(:,1:length(node.input))';1*AXT(:,length(node.input)+1:end)']);
AXT = [AXT;zeros(1,length(a))];
% Solve system equation
pvec = Am\AXT;
state.AXT = sparse(AXT);
state.C = pvec;
state.P = state.ATX*pvec;
end
