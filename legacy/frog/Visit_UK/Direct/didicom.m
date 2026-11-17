name = [date,'-Co1'];
limit =lm-1;
cS = writeDicom(coSave(:,1:limit),limit,size(ROI),time,name);
% name = [date,'-Coarse'];
% cS = writeDicom(cSave,lm,size(Lores),time,name);
% name = [date,'-Art'];
% cS = writeDicom(cASave,lm,Grid,t,name);
% name = [date,'-Vein'];
% cS = writeDicom(cVSave,lm,Grid,t,name);

function cS = writeDicom(C,lm,Sise,t,name)
Amax = full(max(C(:)));
Amin = full(min(C(:)));
lm=lm-1;
start = fix(now);
folder = ['Result/',name];
[~,~,~] = mkdir(folder);
cCSave = C;
% cCSave = int16(reshape(0.1*(2^16)/(Amax-Amin)*full(C(:,1:lm)),Sise(1)*Sise(2),lm));
i=1;
% cS=int16(reshape(0.1*(2^16)/(Amax-Amin)*full(cCSave(:,i)),Sise));
cS=int16(reshape(1000*full(cCSave(:,i)),Sise));
dicomwrite(cS,[folder '/',sprintf('%04d', i),'.dcm']);
info = dicominfo([folder '/',sprintf('%04d', i),'.dcm']);
info.PixelRepresentation = 1;
info.InstanceNumber = 1;
info.Modality = 'MR';
info.RescaleType = 'US';
info.ContentDate = datestr(now, 'YYYY.MM.DD');
% info.RescaleSlope = 10*(Amax-Amin)/(2^16);
info.RescaleSlope = 1/1000;
info.RescaleIntercept = 0;%(Amax-Amin)/(2^16);
info.LargestImagePixelValue = Amax*10;
for i=1:lm
waktu = datestr(start + t(i)/(60*60*24), 'HH:MM:SS.fff');
info.AcquisitionTime = waktu;
info.ContentTime = waktu;
info.SeriesNumber = i;
info.SeriesTime = waktu;
% cS = int16(reshape(0.1*(2^16)/(Amax-Amin)*full(cCSave(:,i)),Sise));
cS = int16(reshape(1000*full(cCSave(:,i)),Sise));
dicomwrite(cS,[folder '/',sprintf('%04d', i),'.dcm'],info, 'CreateMode', 'copy','CompressionMode','None');
end
end