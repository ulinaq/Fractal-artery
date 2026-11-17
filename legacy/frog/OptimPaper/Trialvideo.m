tic;
Tt = 20;
dt = 0.0001;
dV = 500; % video data iteration interval
TL = 0 : dt : Tt;
Tc = numel(TL);
c = zeros(Ca+vnode.n+2*N,1);
cSave = zeros(Ca+vnode.n+2*N,round(Tc/dV));
% Initial concentration in arterial root
C0 = 1;
t0 = 0;
%fixed parameters
a  = 3.0;   b  = 1.5;
%change from minutes to seconds
t  = TL*60; t0 = t0*60; dt=dt*60;
%get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
AIF = AIF.*(t>=t0);
c(1:Ca,1) = AIF(1);
%% Coupling between arterial terminal nodes and capillary
Artin = node.num(node.isterminal);
qq = []; qa = []; cell = []; % temporary variable
ATC = sparse(zeros(N,Ca));
for i=1:Ca
ATC = ATC + sparse(Grid.src.cell(:,i),i, max(Grid.src.rate{i}*q(i),0), N,Ca);
end
%% Combine block matrices
A = [sparse(zeros(vnode.n+2*N,Ca)),speye(vnode.n+2*N)] + dt/Grid.volume*[prm.Por*ATC,AC,spdiags(-1*min(perf(:),0),0, N, N),sparse(zeros(N,vnode.n));...
sparse(zeros(N,Ca)),spdiags(max(perf(:),0),0, N, N),VC,VTC*prm.Por;...
sparse(zeros(vnode.n,(N+Ca))),VCT/prm.Por,VT];
% Color scale
clim = [0,0.25];
% Open movie
folder = ['Result'];
[~,~,~] = mkdir(folder);
str = date;
pathsave = [folder '/' str '.avi'];
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 100;
open(v);
H = figure('Position',[200,100,1000,400]);
% time delay in the vessels
vT = round(Vascular(node,vnode,segment,x)/dt);
% iteration
j = 1;  % Counter to save the data
for i = 2 : numel(TL)
c(Ca+1:end) = A*c(:);
c(1:Ca) = AIF(max([i-vT(node.isterminal)],1));
if mod(i,dV)==0
figure(H);
subplot(1,3,1);imagesc(reshape(c(Ca+1:Ca+N),Grid.Nx,Grid.Ny),clim);axis image;axis off;title('Arterial compartment');
subplot(1,3,2);imagesc(reshape(c(Ca+N+1:Ca+2*N),Grid.Nx,Grid.Ny),clim);axis image;axis off;title('Venous compartment');
subplot(1,3,3);imagesc(reshape(c(Ca+1:Ca+N)+c(Ca+N+1:Ca+2*N),Grid.Nx,Grid.Ny),clim);axis image;axis off;title('Combined');
%    show(Cmat{1}(:,:,:,k),1)
frame = getframe(gcf);
writeVideo(v,frame);
cSave(:,j)=c(:);
j=j+1;
end
end
close(v);
save([folder '/' str '_data.mat'],'cSave');
toc

function vT = Vascular(node,vnode,segment,x)
Co = node.n + vnode.n;
VG = diag(ones(node.n,1));
vb = zeros(node.n,1);
for i = 1 : segment.n
    VG(segment.end(i),[segment.init(i) segment.end(i)]) = [-1 1]*x(Co+i);
    vb(segment.end(i)) = segment.len(i)*pi*segment.rad(i)^2;
end
vT = VG\vb;
end