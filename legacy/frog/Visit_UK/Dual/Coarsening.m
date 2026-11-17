tic;
Tt = 5;
dt = 0.00001;
dV = 100; % video data iteration interval
dn = 30;    % Coarsening factor
TL = 0 : dt : Tt;
Tc = numel(TL);
c = zeros(Co+Grid.N+NAs+NVs,1);
% cSave = sparse(Co+Grid.N+NAs+NVs,round(Tc/dV));
cSave = sparse(round(Grid.Nx/dn)*round(Grid.Ny/dn),round(Tc/dV));
% Initial concentration in arterial root
C0 = 1;
t0 = 0;
% fixed parameters
a  = 3.0;   b  = 1.5;
% change from minutes to seconds
t  = TL*60; t0 = t0*60; dt=dt*60;
% get the AIF
AIF = C0*(t-t0).^a.*exp(-(t-t0)./b);
AIF = AIF.*(t>=t0);
c(1:2,1) = AIF(1);

clim = [0,0.5*max(AIF)];
% Open movie
folder = ['Result'];
[~,~,~] = mkdir(folder);
str = date;
pathsave = [folder '/' str '_Coarse.avi'];
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 100;
open(v);
H = figure('Position',[200,50,800,600]);

for i=1:size(cSave,2)
c = cSave(:,i);
cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny);
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cA(segment.distrind{i}{j})=c(Co+Grid.N+segment.Cnum{i}(j));
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=c(Co+Grid.N+NAs+vsegment.Cnum{i}(j));
    end
end
for i=1:node.n
    cA(node.ind{i})=c(i);
end
for i=1:vnode.n
    cV(vnode.ind{i})=c(node.n+Grid.N+i);
end
cC = sparse(reshape(c(node.n+1:node.n+Grid.N),Grid.Nx,Grid.Ny));
I = cA+cV+cC;
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
 
figure(H);
imagesc(Lores,clim);axis image;axis off; title('Coarse');daspect([1 1 1])
frame = getframe(gcf);
writeVideo(v,frame);
cCoarse(:,i)=Lores(:);
end
close(v);

i=1;
cS=reshape(full(cSave(:,i)),21,17);
dicomwrite(cS,[folder '/Coarse/im',sprintf('%03d', i),'.dcm']);
info = dicominfo([folder '/Coarse/im',sprintf('%03d', i),'.dcm']);
for i=1:300
info.AcquisitionTime = dt*dV*i;
cS=reshape(full(cSave(:,i)),21,17);
dicomwrite(cS,[folder '/Coarse/im',sprintf('%03d', i),'.dcm'],info);
end
cS=cCombSave(:,1:300);
cS=full(cS(:));
save([folder '/Combdata.mat'],'cS');
save([folder '/' str '_Coarse.mat'],'cCoarse');
toc

