for k = 1:numel(time)
    cS = full(reshape(cCombSave(:,k),Grid.Nx,Grid.Ny));
    writeDicom(cS,k,Grid,time,k,subfolder(4,:));
end

function writeDicom(C,lm,Grid,t,k,name,info)
info.AcquisitionTime = t(k);
dicomwrite(C,[name '/im',sprintf('%04d', lm),'.dcm'],info);
end