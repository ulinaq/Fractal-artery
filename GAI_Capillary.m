function [Xr,Grid,node,vnode] = GAI_Capillary(Grid,prm,node,segment,vnode,vsegment)
global Amat cellc 

Nx = Grid.Size(1);  Ny = Grid.Size(2); Nz = Grid.Size(3); N = Nx*Ny*Nz;

%% Define input and output sources in darcy domain
in = node.junction==0;
node.in = in;
out = vnode.junction==0;
vnode.out = out;
DomainLimit = [-Nx/2 Nx/2 0 Ny]; % xmin, xmax, ymin ymax
Reps = 4;   % Epsilon radii for dirac

X = ceil((node.XData(in)- DomainLimit(1))./Grid.h(1));
Y = ceil((node.YData(in)- DomainLimit(3))./Grid.h(2));
Xout = floor((vnode.XData(out)-DomainLimit(1))./Grid.h(1));
Yout = floor((vnode.YData(out))./Grid.h(2));

if max(Xout)>=Nx | max(Yout)>=Ny | min(Xout)<=0 | min(Yout)<=0 ...
        | max(X)>=Nx | max(Y)>=Ny | min(X)<=0 | min(Y)<=0
    Xr.P=NaN; Trans=NaN; Xr.pvec=NaN;
    return;
end

node.input = sub2ind([Nx Ny],X,Y);
vnode.output = sub2ind([Nx Ny],Xout,Yout);

% if any(node.input==vnode.output)
%     Xr.P=NaN; Trans=NaN; Xr.pvec=NaN;
%     return;
% end
%%
A = Amat;
xcell= cellc(:,1);
ycell= cellc(:,2);
zcell= cellc(:,3);
A(N+1,1) = 1;
a = [node.input,vnode.output];
z = zeros(Nx,Ny);
ar = zeros(N,1);
zn = [];
Grid.src.cell = [];
Grid.src.rate = [];
Xr.pvec = [];
for i=1:length(a)
    % Create circle region for spreading epsilon
    z=(Reps+1)^2-(xcell-xcell(a(i))).^2-(ycell-ycell(a(i))).^2;
    area=(xcell-xcell(a(i))).^2+(ycell-ycell(a(i))).^2<=Reps^2;
    Grid.src.cell = [Grid.src.cell; Grid.cell.num(area)];
    ar = ar+area;
    zn=z(area)*prod(Grid.Size)/sum(z(area)*prod(Grid.Size));
    Grid.src.rate{i} = zn;
    q=zeros(N+1,1);   % without source as default
    q(area)=zn;
    q(N+1) = 0;       % dirichlet BC set 0 at corner (0,0)
    % Solve system equation
    Xr.pvec(:,i) = A\q;
end

if any(ar>1)
    Xr.P=NaN; Trans=NaN; Xr.pvec=NaN;
    return;
end

Xr.P=Xr.pvec([node.input,vnode.output],:);

%% Calculate pressure in nodes (Update April 26, 2018)
x = Tree_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% [node,segment,x] = Tree_nonlinEq(node,prm,segment,vnode,vsegment,P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:Co;
q = [x([Co+seg(node.in)-1]);-1*x([Cv+seg(vnode.out)-1])];
Pmat = x(Co)+Xr.pvec*q;
Pmat = reshape(Pmat,Grid.Size(1),Grid.Size(2));
Xr.pressure = Pmat;
V = Flux(Pmat,Grid,T,q);
% MinF = norm((V.Ti-mean(V.Ti)),2);
% MinF = max(V.Ti);
% MinF = std(V.Ti);
% MinF = max(V.Ti)-min(V.Ti);
MinF = x(node.n);
% clear Co Cv q seg Pmat

%% Calculate Flux
function V = Flux(Pmat,Grid,Trans,q)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
       
   V.y = zeros(Nx,Ny+1);
   V.x = zeros(Nx+1,Ny);;
   V.y(:,2:Ny) = (Pmat(:,1:Ny-1)-Pmat(:,2:Ny)).*Trans.y(:,2:Ny);
   V.x(2:Nx,:) = (Pmat(1:Nx-1,:)-Pmat(2:Nx,:)).*Trans.x(2:Nx,:);

   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.u = sqrt(V.qx.^2+V.qy.^2);
   u = reshape(V.u,N,1);
   V.range = max(u)-min(u);
   
   qq=[];
for i=1:length(q)
    qq=[qq; Grid.src.rate{i}*q(i)];
end

qi = sparse(Grid.src.cell,1, qq, N,1);
pv = reshape(Grid.volume.*Grid.poro,N,1);
% define neighbor
num = reshape([1:N],Nx,Ny);
Yin = [zeros(Nx,1) num];
Yout = [num zeros(Nx,1)];
Xin = [zeros(1,Ny); num];
Xout = [num; zeros(1,Ny)];
face.neighbors = [reshape(Xin,[],1), reshape(Xout,[],1);reshape(Yin,[],1), reshape(Yout,[],1)];
face.flux = [reshape(V.x,[],1);reshape(V.y,[],1)];
ind = ~any(face.neighbors==0,2);
n  = double(face.neighbors(ind,:));

out = min(face.flux(ind), 0);
in  = max(face.flux(ind), 0);

% Cell wise total inflow
inflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);

% diagonal 
d = inflow + max(qi,0);

% Inflow flux matrix
A  = sparse(n(:,2), n(:,1),  in, N, N)...
   + sparse(n(:,1), n(:,2), -out, N, N);
A = -A + spdiags(d, 0, N, N);

% Solve
V.Ti = A\pv;
  
end

end