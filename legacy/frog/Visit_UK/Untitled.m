tic;
Tt = 5;
dt = 0.00002;
TL = 0 : dt : Tt;
Tc = numel(TL);
c = zeros(Co+Grid.N+NAs+NVs,1);
% Initial concentration in arterial root
C0 = 1;
t0 = 0.0005;
%fixed parameters
a  = 8;   b  = 0.5;
%change from minutes to seconds
t  = TL*60; t0 = t0*60; dt=dt*60;
dV = round(0.3/dt); % video data iteration interval
dn = 20;    % Coarsening factor
%get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
AIF = AIF.*(t>=t0);
c(1:2,1) = AIF(1);
%% Combine block matrices
A = speye(Co+Grid.N+NAs+NVs) + dt/Grid.volume*[AT,ACT,sparse(node.n,vnode.n),ATS,sparse(node.n,NVs);...
    ATC,AC,VTC,sparse(N,NAs+NVs);...
    zeros(vnode.n,node.n),VCT,VT,sparse(vnode.n,NAs),VTS;...
    AST,sparse(NAs,N+vnode.n),AS,sparse(NAs,NVs);...
    sparse(NVs,N+node.n),VST,sparse(NVs,NAs),VS];
% Color scale
clim = [0,0.005*max(AIF)];
% Open movie
folder = ['Result/',date,'c/'];
[~,~,~] = mkdir(folder);
% subfolder = [folder,'/A';folder,'/V';folder,'/C';folder,'/I';folder,'/L'];
% for i =1:5
%     [~,~,~] = mkdir(subfolder(i,:));
% end
pathsave = [folder '/Coarse.avi'];
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 100;
open(v);
H = figure('Position',[100,100,400,500]);
% iteration
lm = 1;  % Counter to save the data
csum = c;
cASave = sparse(Grid.N,round(0.3252*Tc/dV));
cVSave = sparse(Grid.N,round(0.3252*Tc/dV));
cCSave = sparse(Grid.N,round(0.3252*Tc/dV));
cCombSave = sparse(Grid.N,round(0.3252*Tc/dV));
% cSave = sparse(round(Grid.Nx/dn)*round(Grid.Ny/dn),round(Tc/dV));
% write Dicom info
%waitbar
f = waitbar(0,'1','Name','Computing Image Contrast',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(f,'canceling',0);
time = [];
for iter = 2 : Tc  
c = max(A*c,0);
c(1:2) = AIF(iter);
csum = csum+c;
% if iter==round(Tc/4)
%     dV = 10*dV;
% end
if mod(iter,dV)==0
    % Check for clicked Cancel button
    if getappdata(f,'canceling')
        break
        delete(f);
        close(v);
    end
    time = [time,t(iter)];
    % Update waitbar and message
    waitbar(iter/numel(TL),f,sprintf([num2str(iter/numel(TL)*100),'  ( ',num2str(toc),' s )']))
    cInt = csum./dV;
    csum = 0;
    cA = zeros(Grid.Nx,Grid.Ny); cV = zeros(Grid.Nx,Grid.Ny);
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cA(segment.distrind{i}{j})=cInt(Co+Grid.N+segment.Cnum{i}(j));
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=cInt(Co+Grid.N+NAs+vsegment.Cnum{i}(j));
    end
end
for i=1:node.n
    cA(node.ind{i})=cInt(i);
end
for i=1:vnode.n
    cV(vnode.ind{i})=cInt(node.n+Grid.N+i);
end
cC = reshape(cInt(node.n+1:node.n+Grid.N),Grid.Nx,Grid.Ny)*prm.Por;
I = Grid.B*0.001 + cA*0.45+cV*0.45+cC;
% I(~Grid.B) = 0;
% Get the means by each dn-by-dn block.
% Lores = blockproc(I, [dn dn], Fun);
Lores = imresize(I,1/dn); 
% figure(1); clf;
% imagesc(Lores,clim);axis image;axis off; title('Coarse');daspect([1 1 1])
figure(H);
% subplot(2,2,1);imagesc(cA,clim);axis image;axis off;title('Arterial tree');
% subplot(2,2,2);imagesc(cV,clim);axis image;axis off;title('Venous tree');
% subplot(2,2,3);imagesc(Lores,clim);axis image;axis off;title('Coarse');
% subplot(2,2,4);
imagesc(I,clim);colormap(bone);axis image;axis off;title('Combined');
frame = getframe(gcf);
writeVideo(v,frame);
% cSave(:,lm)=Lores(:);
cASave(:,lm)=cA(:);
cVSave(:,lm)=cV(:);
cCSave(:,lm)=cC(:);
cCombSave(:,lm)=I(:);
lm=lm+1;
end
end
close(v);
delete(f);
toc

function Lores = Rescal(I,dn)
 NS=size(I);
 for k=1:round(NS(1)/dn)
     if k==round(NS(1)/dn)
             for l=1:round(NS(2)/dn)
        if l==round(NS(2)/dn)
            kn = dn*(k-1)+1;
            ln = dn*(l-1)+1;
            Sample=I(kn:end,ln:end);
            Lores(k,l)=mean(Sample(:)); 
        else
        kn = dn*(k-1)+1;
        ln = dn*(l-1)+1;
        Sample=I(kn:end,ln:ln+dn-1);
        Lores(k,l)=mean(Sample(:));
        end
             end
     else
     for l=1:round(NS(2)/dn)
        if l==round(NS(2)/dn)
            kn = dn*(k-1)+1;
            ln = dn*(l-1)+1;
            Sample=I(kn:kn+dn-1,ln:end);
            Lores(k,l)=mean(Sample(:)); 
        else
        kn = dn*(k-1)+1;
        ln = dn*(l-1)+1;
        Sample=I(kn:kn+dn-1,ln:ln+dn-1);
        Lores(k,l)=mean(Sample(:));
        end
     end
     end
 end
end