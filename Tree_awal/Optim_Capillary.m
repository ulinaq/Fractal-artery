function [Xr,Grid,node,vnode] = Optim_Capillary(node, vnode)
global perm Amat cellc Size h 

Grid.Size = Size;  % [Nx Ny Nz]
Grid.h= h;
Grid.cell.N = prod(Grid.Size);
Grid.cell.num = [1:1:Grid.cell.N]'; 
Grid.K = perm;
Grid.poro = ones(Grid.Size);
Grid.volume = ones(Grid.Size)*prod(Grid.h);
Nx = Grid.Size(1);  Ny = Grid.Size(2); Nz = Grid.Size(3); N = Nx*Ny*Nz;


%% Define input and output sources in darcy domain
in = node.junction==0;
node.in = in;
out = vnode.junction==0;
vnode.out = out;
DomainLimit = [-Nx/2 Nx/2 0 Ny]; % xmin, xmax, ymin ymax
Reps = 4;   % Epsilon radii for dirac

X = ceil((node.XData(in)- DomainLimit(1))./Grid.h(1));
Y = ceil((node.YData(in))./Grid.h(2));
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
b = [node.input,vnode.output];
if isempty(b) | min(b)<=0
    Xr.P=NaN; Trans=NaN; Xr.pvec=NaN;
    return;
end
a = uint64(b);
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

% if any(ar>1)
%     Xr.P=NaN; Trans=NaN; Xr.pvec=NaN;
%     return;
% end

Xr.P=Xr.pvec(a,:);

end