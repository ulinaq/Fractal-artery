%% Draw ROIs in domain
figure();
redC = Froi{6}+Froi{1}+Froi{2}+0.2*Froi{9}-0.2*Froi{8}-0.2*Froi{7}+0.8*Froi{10}+Froi{11}-Froi{5};
greenC = Froi{7}+0.7*Froi{3}+0.5*Froi{5}+Froi{2}-0.2*Froi{8}-0.1*Froi{6}+0.2*Froi{9}+0.4*Froi{10}-0.12*Froi{11};
blueC = Froi{8}+Froi{1}+0.5*Froi{4}+0.8*Froi{5}+0.2*Froi{9}-0.02*Froi{6}-0.2*Froi{7}-0.2*Froi{2}+0.5*Froi{6};
rgbImage = cat(3, redC,greenC,blueC);
ima = logical(full(maXc));
ima = imcomplement(ima);
imshow(ima);
hold on
im = imshow(rgbImage);im.AlphaData = 0.5*max(rgbImage,[],3);

figure();
redC = Froi{6}+Froi{1}+Froi{2}+0.2*Froi{9}-0.2*Froi{8}-0.2*Froi{7}+0.8*Froi{10}+Froi{11}-Froi{5};
greenC = Froi{7}+0.7*Froi{3}+0.5*Froi{5}+Froi{2}-0.2*Froi{8}-0.1*Froi{6}+0.2*Froi{9}+0.4*Froi{10}-0.12*Froi{11};
blueC = Froi{8}+Froi{1}+0.5*Froi{4}+0.8*Froi{5}+0.2*Froi{9}-0.02*Froi{6}-0.2*Froi{7}-0.2*Froi{2}+0.5*Froi{6};
rgbImage = cat(3, redC,greenC,blueC);
ima = logical(full(maXc));
ima = imcomplement(ima);
imshow(ima);
hold on
im = imshow(rgbImage);im.AlphaData = 0.5*max(rgbImage,[],3);

% Draw vessels
figure();
load('frog_trees.mat')
for i=1:Connect.n
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    cO(Connect.distrind{i}{j})=1;
    end
end

redC = ones(size(cO))-tree_v{1, 1}.bw-cO;
greenC = ones(size(cO))-cO-tree_a{1, 1}.bw-tree_v{1, 1}.bw;
blueC = ones(size(cO))-tree_a{1, 1}.bw-cO;
rgbImage = cat(3, redC,greenC,blueC);
im = imshow(rgbImage);%im.AlphaData = 0.5*max(rgbImage,[],3);
hold on
delete redC blueC greenC;
redC = Froi{6}+Froi{1}+Froi{2}+0.2*Froi{9}-0.2*Froi{8}-0.2*Froi{7}+0.8*Froi{10}+Froi{11}-Froi{5};
greenC = Froi{7}+0.7*Froi{3}+0.5*Froi{5}+Froi{2}-0.2*Froi{8}-0.1*Froi{6}+0.2*Froi{9}+0.4*Froi{10}-0.12*Froi{11};
blueC = Froi{8}+Froi{1}+0.5*Froi{4}+0.8*Froi{5}+0.2*Froi{9}-0.02*Froi{6}-0.2*Froi{7}-0.2*Froi{2}+0.5*Froi{6};
rgbImage = cat(3, redC,greenC,blueC);
im = imshow(rgbImage);im.AlphaData = 0.5*max(rgbImage,[],3);

