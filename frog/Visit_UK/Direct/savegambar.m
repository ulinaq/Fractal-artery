folder = ['Result/Gambar/'];
[~,~,~] = mkdir(folder);
for i=3:6
pathsave = [folder,'/Plot',datestr(now,'HHMMSS'),num2str(i),'.png'];
figure(i)
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
end
%% Plot multiple boxplot

TreadmillData = randi([20,200],69,6);
Speeds = {'1.5mph' '2.5mph' '3.5mph' '4.5mph' '5.5mph' '6.5mph'};
DeviceColors = {'r' 'g' 'c' [0.5 0 0.5] 'b' [1 0.5 0]};
Pedometer1 = TreadmillData(1:7:end,:);
Pedometer2 = TreadmillData(2:7:end,:);
Pedometer3 = TreadmillData(3:7:end,:);
Pedometer4 = TreadmillData(4:7:end,:);
Pedometer5 = TreadmillData(5:7:end,:);
Pedometer6 = TreadmillData(6:7:end,:);

GroupedData = {Pedometer1 Pedometer2 Pedometer3 Pedometer4 Pedometer5 Pedometer6}; 

legendEntries = {'dev1' 'dev2' 'dev3' 'dev4' 'dev5' 'dev6'};

N = numel(GroupedData);
delta = linspace(-.3,.3,N); %// define offsets to distinguish plots
width = .2; %// small width to avoid overlap
cmap = hsv(N); %// colormap
legWidth = 1.8; %// make room for legend

figure;
hold on;

for ii=1:N %// better not to shadow i (imaginary unit)
        labels = Speeds; %// center plot: use real labels
    boxplot(GroupedData{ii},'Color', DeviceColors{ii}, 'boxstyle','filled', ...
        'position',(1:numel(labels))+delta(ii), 'widths',width, 'labels',labels)
        %// plot filled boxes with specified positions, widths, labels
    plot(NaN,1,'color',DeviceColors{ii}); %// dummy plot for legend
end
xlabel('Speed'); ylabel('Step Count'); grid on;
xlim([1+2*delta(1) numel(labels)+legWidth+2*delta(N)]) %// adjust x limits, with room for legend

legend(legendEntries);


% % end
% msg = ['Saving ' pathsave]; disp(msg);
% 
% % for i = 1:11
% %     pathsave = [folder,'/Vol',datestr(now,'ddmmyy_HHMMSSFFF'),'.png']; 
% %     if mod(i,2)==0
%         pathsave = [folder,'/AIF3',datestr(now,'ddmmyy_HHMMSSFFF'),'.png']; 
% %     else
% %         pathsave = [folder,'/AIF1',datestr(now,'ddmmyy_HHMMSSFFF'),'.png']; 
% %     end
% figure(i)
% set(gcf, 'Color', 'w');
% export_fig(pathsave,'-r300');
% close(i)
% % end

% b(1,:) = b(1,:)*0.55;
% for i = 1:11
%    Fv = b(:,i); 
% HM = figure('name',['Model Fitting'],'Position',[100 100 420 300]);
% plot([min(Fv),max(Fv)],[min(Fv),max(Fv)],'k','LineWidth',1);title(['Blood volume comparison']);
% hold on
% plot(Fv(1),Fv(3),'b*','LineWidth',2);
% plot(Fv(1),Fv(2),'m*','LineWidth',2);
% xlabel('Blood volume model [ml/100ml]')
% ylabel(['Blood volume prediction [ml/100ml]'])
% legend({'Identity line','2CM Uptake','2CM Exchange'},'Location','northwest');
% set(gcf, 'Color', 'w');
% end

%% Plot fitting all data AIFs Handpicked ROIs
% load('DataROI.mat');
% as2 = 
% lw = 2; ms = 15;
% HM = figure('name',['Model Fitting'],'Position',[100 100 1500 800]);
% figure(HM)
% plot([0,12],[0,0],'k','LineWidth',lw);title(['Plasma Flow estimation for local and global AIF']);
% hold on
% b = bar(1:11,(as2-1)*100);
% b(1).FaceColor = 'r'; 
% b(2).FaceColor = 'g';
% b(3).FaceColor = [0.1 0.8 0];
% b(4).FaceColor = 'c';
% b(5).FaceColor = [0.1 0.8 0.8];
% b(6).FaceColor = 'b';
% b(7).FaceColor = [0.1 0.1 0.7];
% b(8).FaceColor = 'm';
% b(9).FaceColor = [0.8 0.1 0.8];
% xlabel('ROI')
% ylabel(['Error [%]'])
% legend({'Real','MS analytic(Local AIF)','MS(Local AIF)','MS(Global AIF)',...
%     '1CM(Local AIF)','1CM(Global AIF)','2CM Uptake(Local AIF)',...
%     '2CM Uptake(Global AIF)','2CM Exchange(Local AIF)',...
%     '2CM Exchange(Global AIF)'},'Location','northwest');
% set(gcf, 'Color', 'w');
% 
%%
%%%%%% Local AIF
load('DataROI.mat');
as = a1(3:end,:);
as(1,:) = a1(5,:);
as(2,:) = a1(6,:);
as(3,:) = a1(4,:);
as(4,:) = a1(3,:);
lw = 2; ms = 15;
HM = figure('name',['Model Fitting'],'Position',[100 100 1500 800]);
figure(HM)
plot([0,12],[0,0],'k','LineWidth',lw);title(['Plasma flow estimation using local AIF']);
hold on
b = bar(1:11,(as-1)*100);
b(1).FaceColor = 'g';
b(2).FaceColor = 'c';
b(3).FaceColor = 'b';
b(4).FaceColor = 'm';
xlabel('ROI')
xticks([1:1:11])
ylabel(['Error [%]'])
legend({'Ground truth data','MS','1CM','2CM Uptake','2CM Exchange'},'Location','northwest');
set(gcf, 'Color', 'w');
%%
%%%%
HM = figure('name',['Model Fitting'],'Position',[100 100 1500 800]);
figure(HM)
plot([0,12],[0,0],'k','LineWidth',lw);title(['MS and 1CM estimation comparison using local and global AIF']);
hold on
as2 = [a1(5,:);a2(4,:);a1(6,:);a2(5,:)]
b = bar(1:11,(as2-1)*100);
b(1).FaceColor = 'g'; 
b(2).FaceColor = [0.360 0.5240 0.1380];
b(3).FaceColor = 'c';
b(4).FaceColor = [0 0.4470 0.7410];

% plot([1:11],a1(2,:),'ro-.','LineWidth',lw,'MarkerSize',ms);
% plot([1:11],(a1(5,:)-1)*100,'go--','LineWidth',lw,'MarkerSize',ms);
% plot([1:11],(a2(4,:)-1)*100,'*:','Color',[0.360 0.5240 0.1380],'LineWidth',lw,'MarkerSize',ms);
% plot([1:11],(a1(6,:)-1)*100,'co--','LineWidth',lw,'MarkerSize',ms);
% plot([1:11],(a2(5,:)-1)*100,'*:','Color',[0 0.4470 0.7410],'LineWidth',lw,'MarkerSize',ms);
% plot([1:11],a1(3,:),'bo--','LineWidth',lw,'MarkerSize',ms);
% plot([1:11],a2(3,:),'b*:','LineWidth',lw,'MarkerSize',ms);
% plot([1:11],a1(4,:),'mo--','LineWidth',lw,'MarkerSize',ms);
% plot([1:11],a2(2,:),'m*:','LineWidth',lw,'MarkerSize',ms);
xlabel('ROI')
xticks([1:1:11])
ylabel(['Error[%]'])
legend({'Real','MS(Local AIF)','MS(Global AIF)',...
    '1CM(Local AIF)','1CM(Global AIF)'},'Location','northwest');
set(gcf, 'Color', 'w');

%%% Plot plasma volume
load('DataBV_BF.mat');
HM = figure('name',['Model Fitting'],'Position',[100 100 1500 800]);
figure(HM)
plot([0,12],[0,0],'k','LineWidth',lw);title(['plasma volume estimation']);
hold on
bv = bvL;
bvL = bv(2:3,:)./bv(1,:);
bv = bvG;
bvG = bv(2:3,:)./bv(1,:);
bs2 = [bvL(1,:);bvL(2,:);bvG(1,:);bvG(2,:)];
b = bar(1:11,(bs2-1)*100);
b(1).FaceColor = 'b';   b(1).EdgeColor = 'b'; 
b(2).FaceColor = 'm';   b(2).EdgeColor = 'm';
b(3).FaceColor = '#6EB5FF'; b(3).EdgeColor = '#6EB5FF';
b(4).FaceColor = '#D5AAFF'; b(4).EdgeColor = '#D5AAFF';
xlabel('ROI')
xticks([1:1:11])
ylim([-25 25])
ylabel(['Error[%]'])
legend({'Real','2CM Uptake(Local AIF)','2CM Exchange(Local AIF)',...
    '2CM Uptake(Global AIF)','2CM Exchange(Global AIF)'},'Location','northwest');
set(gcf, 'Color', 'w');



% %%
% HM = figure('name',['Model Fitting'],'Position',[100 100 1500 800]);
% figure(HM)
% plot([0,12],[1,1],'k','LineWidth',1);title(['Blood Flow estimation using local AIF']);
% hold on
% plot([1:11],a1(2,:),'ro:','LineWidth',2,'MarkerSize',8);
% plot([1:11],a1(5,:),'go--','LineWidth',2,'MarkerSize',8);
% plot([1:11],a1(6,:),'co-.','LineWidth',2,'MarkerSize',8);
% plot([1:11],a1(3,:),'bo-.','LineWidth',2,'MarkerSize',8);
% plot([1:11],a1(4,:),'mo--','LineWidth',2,'MarkerSize',8);
% plot([1:11],a2(4,:),'gx','LineWidth',2,'MarkerSize',8);
% plot([1:11],a2(5,:),'cx','LineWidth',2,'MarkerSize',8);
% plot([1:11],a2(3,:),'bx','LineWidth',2,'MarkerSize',8);
% plot([1:11],a2(2,:),'mx','LineWidth',2,'MarkerSize',8);
% xlabel('ROI')
% ylabel(['Normalized Blood flow'])
% % legend({'Real','MS analytic(Local AIF)','MS(Local AIF)',...
% %     '1CM(Local AIF)','2CM Uptake(Local AIF)','2CM Exchange(Local AIF)',...
% %     'MS(Global AIF)','1CM(Global AIF)','2CM Uptake(Global AIF)',...
% %     '2CM Exchange(Global AIF)'},'Location','northwest');
% set(gcf, 'Color', 'w');

%% 
% b =zeros(size(Bflow));
% for i=1:length(Bflow)
% b(1:5,i)=Bflow(1:5,i)/Bflow(1,i);
% b(6:8,i)=Bflow(6:8,i)/Bflow(6,i);
% end