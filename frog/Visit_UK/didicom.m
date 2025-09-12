name = [date,'-Co'];
limit = 2000;
cS = writeDicom(cCombSave(:,1:limit),limit,size(I),time,name);
name = [date,'-Coarse'];
[ha, pos] = tight_subplot(4, 4, [.01 .03],[.1 .01],[.01 .01])
clim = [0,0.02*maxAIF];
a = 130+[0,5,10,15,20,50,75,100,150,250,300,450,600,800,1000,1200];

for i =1:16
cS=reshape(full(cCombSave(:,a(i))),Grid.Nx,Grid.Ny);
figure(1);
clf
imagesc(cS,clim);axis image;axis off; colormap('bone');
export_fig(num2str(i),'-r300'); 
end
set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','')
function cS = writeDicom(C,lm,Sise,t,name)
Amax = full(max(C(:)));
Amin = full(min(C(:)));
lm=lm-1;
start = fix(now);
folder = ['Result/',name];
[~,~,~] = mkdir(folder);
cCSave = int16(reshape(0.6*(2^16)/(Amax-Amin)*full(C(:,1:lm)),Sise(1)*Sise(2),lm));
i=1;
cS=reshape(full(cCSave(:,i)),Sise);
dicomwrite(cS,[folder '/',sprintf('%04d', i),'.dcm']);
info = dicominfo([folder '/',sprintf('%04d', i),'.dcm']);
info.PixelRepresentation = 1;
info.InstanceNumber = 1;
info.Modality = 'MR';
info.RescaleType = 'US';
info.ContentDate = datestr(now, 'YYYY.mm.DD');
info.RescaleSlope = 100*(Amax-Amin)/(2^16);
info.RescaleIntercept = 0;%(Amax-Amin)/(2^16);
info.LargestImagePixelValue = Amax*10;
for i=1:lm
waktu = datestr(start + t(i)/(60*60*24), 'HH:MM:SS.fff');
% waktu = start + t(i)/(60*60*24);
info.AcquisitionTime = waktu;
info.ContentTime = waktu;
info.SeriesNumber = i;
info.SeriesTime = t(i)/(60*60*24);
cS = reshape(full(cCSave(:,i)),Sise);
dicomwrite(cS,[folder '/',sprintf('%04d', i),'.dcm'],info, 'CreateMode', 'copy','CompressionMode','None');
end
end