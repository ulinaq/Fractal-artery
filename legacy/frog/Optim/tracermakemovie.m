function [v] = tracermakemovie(c,Grid)
Ca=60; N = Grid.N;
% Combine the two compartments
C = c(Ca+1:Ca+N,:) + c(Ca+N+1:Ca+2*N,:);

% Color scale
clim = [0,max(C(:))/10];

% Open movie
folder = ['Result'];
[~,~,~] = mkdir(folder);
str = date;
pathsave = [folder '/' str '.avi'];        
msg = ['Saving ' pathsave];
disp(msg);
v = VideoWriter(pathsave);
v.FrameRate = 120;
open(v);

nt = size(C,2);
H = figure('Position',[200,100,1000,400]);
for k = 1:nt    
   figure(H);
   subplot(1,3,1);imagesc(reshape(c(Ca+1:Ca+N,k),Grid.Nx,Grid.Ny),clim);axis image;axis off;title('Arterial compartment');
   subplot(1,3,2);imagesc(reshape(c(Ca+N+1:Ca+2*N,k),Grid.Nx,Grid.Ny),clim);axis image;axis off;title('Venous compartment');
   subplot(1,3,3);imagesc(reshape(C(:,k),Grid.Nx,Grid.Ny),clim);axis image;axis off;title('Combined');
%    show(Cmat{1}(:,:,:,k),1)
   frame = getframe(gcf);
   writeVideo(v,frame);
%    pause
end
close(v);