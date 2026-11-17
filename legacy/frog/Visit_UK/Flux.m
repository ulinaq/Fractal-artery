function V = Flux(prm,Pa,Grid,Trans,q,node,vnode,segment,vsegment,x)
Ny=Grid.Ny; Nx=Grid.Nx; Nz=Grid.Nz; N=Nz*Ny*Nx;
Aa = Flow(Pa,Grid,Trans.ay,Trans.ax);
[VG,vb] = Vascular(node,vnode,segment,x);
VL = zeros(N,node.n);
qq=[];
cell=[];
term = node.num(node.isterminal);
for i=1:length(node.input)
    qq=[qq; Grid.src.rate{i}*q(i)];
    cell = [cell;Grid.src.cell{i}];
    VL(Grid.src.cell{i},term(i)) = -Grid.src.rate{i}*q(i);
end
qa = sparse(cell,1, qq, N,1);
% diagonal from sources
da = Aa.inflow + max(qa,0);
pa = reshape(Grid.volume*Grid.poro,N,1)/4;
[da,pa,Aa] = maxtof(pa,max(qa,0),Aa,da);
% Create matrix artery
AA  = sparse(Aa.n(:,2), Aa.n(:,1),  Aa.in, N, N)...
   + sparse(Aa.n(:,1), Aa.n(:,2), -Aa.out, N, N);
AA = -AA + spdiags(da, 0, N, N);

A = [VG,zeros(node.n,N);VL,AA];
pp =[vb;pa];
% Solve
V.Ti = A\pp;
[Vol,Flo] = Comp(prm,Grid,node,vnode,segment,vsegment,Aa,x);
V.Ta = reshape(V.Ti(node.n+1:node.n+Grid.N),Grid.Nx,Grid.Ny);
V.Vol = Vol;    V.Flow = Flo;
function [Vol,Flo] = Comp(prm,Grid,node,vnode,segment,vsegment,Aa,x)
cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny);
fA = sparse(Grid.Nx,Grid.Ny); fV = sparse(Grid.Nx,Grid.Ny);
Co = node.n + vnode.n;  Cv = Co + segment.n;
for i=1:segment.n
    ns = length(segment.Cnum{i});
    fA(node.ind{segment.end(i)})=x(Co+i);
    for j = 1:ns
    cA(segment.distrind{i}{j})=1;
    fA(segment.distrind{i}{j})=x(Co+i);
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    fV(vnode.ind{vsegment.end(i)})=x(Cv+i);
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=1;
    fV(vsegment.distrind{i}{j})=x(Cv+i);
    end
end
for i=1:node.n
    cA(node.ind{i})=1;
end
for i=1:vnode.n
    cV(vnode.ind{i})=1;
end
Vol = (Grid.B*prm.Por+cA*0.4+cV*0.4)*0.55;
Vol(~Grid.B) = 0;
Flo = (reshape(Aa.inflow,Grid.Nx,Grid.Ny)+fA+fV)*0.55*60/100/Grid.volume;
Flo(~Grid.B) = 0;
end

function [VG,vb] = Vascular(node,vnode,segment,x)
Co = node.n + vnode.n;
VG = diag(ones(node.n,1));
vb = zeros(node.n,1);
for i = 1 : segment.n
    VG(segment.end(i),[segment.init(i) segment.end(i)]) = [-1 1]*x(Co+i);
    vb(segment.end(i)) = segment.len(i)*pi*segment.rad(i)^2;
end
% vT = VG\vb;
end        

function [da,pa,Aa] = maxtof(pa,qa,Aa,da)
% Handling of T -> inf (Boundary region included) :
maxTOF = 50*sum(pa)/sum(qa);
% Find cells that reach max TOF locally
maxIn = max(da);
aboveMax = full(pa./ (max(da, eps*maxIn)) ) > maxTOF;
% Set aboveMax-cells to maxTOF
da(aboveMax)  = maxIn;
pa(aboveMax) = maxIn*maxTOF;
% remove upstream connections
Aa.in(aboveMax(Aa.n(:,2)))  = 0;
Aa.out(aboveMax(Aa.n(:,1))) = 0;
end
function An = Flow(Pa,Grid,Trany,Tranx)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
   V.y = zeros(Nx,Ny+1);
   V.x = zeros(Nx+1,Ny);
   V.y(:,2:Ny) = (Pa(:,1:Ny-1)-Pa(:,2:Ny)).*Trany(:,2:Ny);
   V.x(2:Nx,:) = (Pa(1:Nx-1,:)-Pa(2:Nx,:)).*Tranx(2:Nx,:);

   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.u = sqrt(V.qx.^2+V.qy.^2);
   u = reshape(V.u,N,1);
   V.range = max(u)-min(u);
   % define neighbor cell
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
An.n = n; An.in = in; An.out = out; An.inflow = inflow;
end
end
