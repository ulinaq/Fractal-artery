% function V = TracerForward(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,vsegment,x)
% Compute tracer indicator in the vascular system.
%
Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
Aa = Flow(Pa,Grid,Trans.ay,Trans.ax);
Av = Flow(Pv,Grid,Trans.vy,Trans.vx);
perf = prm.alpha*(Pa-Pv);

%% Arterial tree matrix
Co = node.n + vnode.n;  Cv = Co + segment.n; Ca = length(node.input);
AT = zeros(node.n);
for i = 1 : segment.n
    if x(Co+i)>0
        AT(segment.end(i),segment.init(i)) = x(Co+i);%/(pi*segment.len(i)*segment.rad(i)^2);
        AT(segment.init(i),segment.init(i)) = -x(Co+i);%/(pi*segment.len(i)*segment.rad(i)^2);
    else
        AT(segment.init(i),segment.end(i)) = x(Co+i);%/(pi*segment.len(i)*segment.rad(i)^2);
        AT(segment.end(i),segment.end(i)) = -x(Co+i);%/(pi*segment.len(i)*segment.rad(i)^2);
    end
end

%% Venal tree
VT = zeros(vnode.n);
for i = 1 : segment.n
    if x(Cv+i)>0
        VT(vsegment.init(i),vsegment.end(i)) = x(Cv+i);%/(pi*vsegment.len(i)*vsegment.rad(i)^2);
        VT(vsegment.end(i),vsegment.end(i)) = -x(Cv+i);%/(pi*vsegment.len(i)*vsegment.rad(i)^2);
    else
        VT(vsegment.end(i),vsegment.init(i)) = x(Cv+i);%/(pi*vsegment.len(i)*vsegment.rad(i)^2);
        VT(vsegment.init(i),vsegment.init(i)) = -x(Cv+i);%/(pi*vsegment.len(i)*vsegment.rad(i)^2);
    end
end

%% Coupling between arterial terminal nodes and capillary
Artin = node.num(node.isterminal);
qq = []; qa = []; cell = []; % temporary variable
ACT = sparse(zeros(node.n,N));
ATC = sparse(zeros(N,node.n));
for i=1:Ca
    if q(i)>0 % flow into capillary
        ATC = ATC + sparse(Grid.src.cell(:,i),Artin(i), max(Grid.src.rate{i}*q(i),0), N,node.n);
        AT(Artin(i),Artin(i)) = - q(i)/(pi*segment.len(Artin(i))*segment.rad(Artin(i))^2);
    else % flow from capillary to artery
        ACT = ACT + sparse(Artin(i),Grid.src.cell(:,i), -1*min(Grid.src.rate{i}*q(i),0), node.n,N);
        qq = [qq; Grid.src.rate{i}*q(i)];
        cell = [cell;Grid.src.cell(:,i)];
    end
end
qa = sparse(cell,1, qq, N,1);

%% Create matrix Arterial compartment
% diagonal matrix = outflow from cells, including perfusion to venal compartment
% outflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);
da = qa + Aa.outflow - max(perf(:),0); % outflow to neighbors, artery, and perfusion to venal comp
% off-diagonal matrix is inflow from neighbors 
AC = sparse(Aa.n(:,2), Aa.n(:,1),  Aa.out, N, N)...
   + sparse(Aa.n(:,1), Aa.n(:,2), -Aa.in, N, N);
AC = (AC + spdiags(da, 0, N, N));

%% Coupling between venous terminal nodes and capillary
Artin = vnode.num(vnode.isterminal);
qq = []; qv = []; cell = []; % temporary variable
VCT = sparse(zeros(vnode.n,N));
VTC = sparse(zeros(N,vnode.n));
for i=Ca+1:Ca+length(vnode.output)
    if q(i)>0
        VTC = VTC + sparse(Grid.src.cell(:,i),Artin(i-Ca), max(Grid.src.rate{i}*q(i),0), N,vnode.n);
        VT(Artin(i-Ca),Artin(i-Ca)) = - q(i)/(pi*vsegment.len(Artin(i-Ca))*vsegment.rad(Artin(i-Ca))^2);
    else
        VCT = VCT + sparse(Artin(i-Ca),Grid.src.cell(:,i), -1*min(Grid.src.rate{i}*q(i),0), vnode.n,N);
        qq = [qq; Grid.src.rate{i}*q(i)];
        cell = [cell;Grid.src.cell(:,i)];
    end
end
qv = sparse(cell,1, qq, N,1);

%% Create matrix Venal compartment
dv = qv + Av.outflow + min(perf(:),0); % outflow to neighbors, vein, and perfusion to arterial comp
% off diagonal
VC  = sparse(Av.n(:,2), Av.n(:,1),  -Av.in, N, N)...
   + sparse(Av.n(:,1), Av.n(:,2), Av.out, N, N);
VC = (VC + spdiags(dv, 0, N, N));
% 
% 
% 
% %% Tracer concentration iteration
% Tt = 1;
% dt = 0.0005;
% TL = 0 : dt : Tt;
% Tc = numel(TL);
% c = zeros(Co+2*N,Tc);
% % Initial concentration in arterial root
% C0 = 1;
% t0 = 0;
% %fixed parameters
% a  = 3.0;   b  = 1.5;
% %change from minutes to seconds
% t  = TL*60; t0 = t0*60; dt=dt*60;
% %get the AIF
% AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
% root = node.num(node.isroot);
% AIF = AIF.*(t>=t0);
% c(root,1) = AIF(1);
% %% Combine block matrices
% % A = speye(Co+2*N) + dt*[AT,ACT,sparse(zeros(node.n,(N+vnode.n)));...
% %     ATC,AC,spdiags(-1*min(perf(:),0),0, N, N),sparse(zeros(N,vnode.n));...
% %     sparse(zeros(N,node.n)),spdiags(max(perf(:),0),0, N, N),VC,VTC;...
% %     sparse(zeros(vnode.n,(N+node.n))),VCT,VT];
% A = speye(Co+2*N) + dt/Grid.volume*[AT,ACT/prm.Por,sparse(zeros(node.n,(N+vnode.n)));...
%     prm.Por*ATC,AC,spdiags(-1*min(perf(:),0),0, N, N),sparse(zeros(N,vnode.n));...
%     sparse(zeros(N,node.n)),spdiags(max(perf(:),0),0, N, N),VC,prm.Por*VTC;...
%     sparse(zeros(vnode.n,(N+node.n))),VCT/prm.Por,VT];
% % iteration
% for i = 2 : numel(TL)
%     c(:,i) = max(A*c(:,i-1),0);
%     c(root,i) = AIF(i);
% end

%% Functions
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
face.neighbors = [reshape(Xin,[],1), reshape(Xout,[],1);reshape(Yin,[],1), reshape(Yout,[],1)]; % face area between 2 cells
face.flux = [reshape(V.x,[],1);reshape(V.y,[],1)]; % flux corresponding to face 
ind = ~any(face.neighbors==0,2);
n  = double(face.neighbors(ind,:));
in = min(face.flux(ind), 0); % negative value, inflow to cell i
out  = max(face.flux(ind), 0); % positive, goes outward

% Cell wise total outflow
outflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);
An.n = n; An.in = in; An.out = out; An.outflow = outflow;
end
% end
