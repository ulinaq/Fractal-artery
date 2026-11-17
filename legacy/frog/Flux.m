function V = Flux(Pmat,Grid,Trans,q,aa)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
   % Compute Fluxes in all faces    
   V.y = zeros(Nx,Ny+1);
   V.x = zeros(Nx+1,Ny);;
   V.y(:,2:Ny) = (Pmat(:,1:Ny-1)-Pmat(:,2:Ny)).*Trans.y(:,2:Ny);
   V.x(2:Nx,:) = (Pmat(1:Nx-1,:)-Pmat(2:Nx,:)).*Trans.x(2:Nx,:);

   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.u = sqrt(V.qx.^2+V.qy.^2);
   u = reshape(V.u,N,1);
   V.range = max(u)-min(u);
   % Find external sources from 
   qq=[];
for i=1:length(q)
    qq=[qq; Grid.src.rate{i}*q(i)];
end

qi = sparse(Grid.src.cell,1, qq, N,1);
pv = reshape(Grid.volume.*Grid.poro,N,1);
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

% diagonal 
d = inflow + max(qi,0);

% Inflow flux matrix
A  = sparse(n(:,2), n(:,1),  in, N, N)...
   + sparse(n(:,1), n(:,2), -out, N, N);
A = -A + spdiags(d, 0, N, N);
% A = A + spdiags(aa,0,Ny*Nx,Ny*Nx);

% Solve
V.Ti = A\pv;
  
end
