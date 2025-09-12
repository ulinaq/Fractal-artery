function [state,Trans,Grid,node,vnode] = Frog_Capillary(node, vnode)
%% Create darcy domain
% Reps = 2;   % Epsilon radii for dirac
A = imread('frog_domain_low.png');
B = im2bw(A);
C = bwperim(B);
c = find(C);
Bz = size(B);
A = 1-B;
A = reshape(A,prod(Bz),1);
% D = ones(6,6);
% Bz = size(D);
% D(3,3) = 0;
% A = 1-D;
% A = reshape(A,prod(Bz),1);
Grid = struct('Nx',[]);
Grid.Nx=Bz(1); Grid.hx=1;
Grid.Ny=Bz(2); Grid.hy=1;
Grid.Nz=1; Grid.hz=1;
Grid.K=B;
Grid.poro = ones(Grid.Nx,Grid.Ny,Grid.Nz);
Grid.volume = ones(Grid.Nx,Grid.Ny,Grid.Nz)*Grid.hx*Grid.hy*Grid.hz;

X = ceil((node.XData(node.isterminal))./Grid.hx);
Y = ceil(node.YData(node.isterminal)./Grid.hy);
Xout = floor((vnode.XData(vnode.isterminal))./Grid.hx);
Yout = floor((vnode.YData(vnode.isterminal))./Grid.hy);

if max(Xout)>=Grid.Nx | max(Yout)>=Grid.Ny | min(Xout)<=0 | min(Yout)<=0 ...
        | max(X)>=Grid.Nx | max(Y)>=Grid.Ny | min(X)<=0 | min(Y)<=0
    state.P=NaN; Trans=NaN; state.pvec=NaN;
    return;
end

node.input = sub2ind([Grid.Nx Grid.Ny],X,Y);
vnode.output = sub2ind([Grid.Nx Grid.Ny],Xout,Yout);
%% Darcy Flow
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
Amat(N+1,c(1)) = 1;
a = [node.input;vnode.output];
% z = zeros(Nx,Ny);
% ar = zeros(Nx,Ny);
zn = [];
% Grid.cell = [1:Nx*Ny]';
Grid.src.cell = [];
Grid.src.rate = [];
state.pvec = [];
for i=1:length(a)
    % Create circle region for spreading epsilon
%     z=(Reps+1)^2-(xcell-xcell(a(i))).^2-(ycell-ycell(a(i))).^2;
%     area=(xcell-xcell(a(i))).^2+(ycell-ycell(a(i))).^2<=Reps^2;
%     ar = ar+area;
    area= [a(i)-2*Nx, a(i)-Nx-1, a(i)-Nx, a(i)-Nx+1, a(i)-2, a(i)-1, a(i), a(i)+1, a(i)+2, a(i)+Nx-1, a(i)+Nx, a(i)+Nx+1, a(i)+2*Nx]';
    zn = [0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989;0.0898876404494382;0.101123595505618;0.0898876404494382;0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989];
%     zn=z(area)*hx*hy/sum(z(area)*hx*hy);
    Grid.src.rate{i} = zn;
    Grid.src.cell = [Grid.src.cell; area];
    q=zeros(N+1,1);   % without source as default
    q(area)=zn;
    % Solve system equation
    state.pvec(:,i) = Amat\q;
end

% ar = reshape(ar,Nx*Ny,1);
% if any(ar>1)
%     state.P=NaN; Trans=NaN; state.pvec=NaN;
%     return;
% end

state.P=state.pvec(a,:);
Trans.y=Tymat;
Trans.x=Txmat;
end

