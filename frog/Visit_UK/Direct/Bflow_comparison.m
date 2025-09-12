Bfl = Bflo;
bpi=[]; bp=[]; g=[];

% warna = [1 0.2 0.3];
% name = ['Blood volume estimation']
% for i = 7:8
warna = [0 0.5 1];
name = ['Plasma flow estimation']
for i = 2:5
    
    for jk = 1:20
        lm(jk) = length(Bfl{jk});
    end
    k = min(lm);
    for jk = 1:20
        b = Bfl{jk};
        temp = b(i,:);
        [tem] = find(temp>100 | temp<0);
        temp(unique(tem))=[];
        temp = (temp-1)*100;
        if length(temp)>200
            temp = temp(randperm(length(temp),200));
        end
        bp = [bp;temp'];
        g = [g;repmat(jk*10*prm.pixel,length(temp),1)];
        bpi(i,jk) = length(temp);
    end
end
    figure();
    plot([0,20],[1,1],'k--','LineWidth',1);title(name);
    hold on
    hd = boxplot(bp,g, 'Whisker', 1.5, 'PlotStyle','compact', 'Symbol','.','ExtremeMode','clip', 'Colors', warna);
    ylim([-100 300])
    ylabel(['Estimation Error [%]'])
    xlabel(['ROI radius [mm]'])
% boxplot(bp,g,'Whisker',2);
% ylim([0 10])
% 
% bp = [b(2,:)'; b(3,:)'; b(4,:)'; b(5,:)'; b(7,:)'; b(8,:)'];
% nm = length(b);
% g = [repmat({'2CMX Flow'},nm,1); repmat({'2CMU Flow'},nm,1); repmat({'MS Flow'},nm,1);...
%     repmat({'1CM Flow'},nm,1); repmat({'2CMX Vol'},nm,1); repmat({'2CMU Vol'},nm,1)];
% %% Plot
% folder = ['Result/Bflow/',date,'/'];
% [~,~,~] = mkdir(folder);
% pathsave = [folder,'/Radius',num2str(j),'0_',datestr(now,'HHMMSS'),'.png'];
% HM = figure('name',['Model Fitting'],'Position',[100 100 700 500]);
% figure(HM)
% 
% 
% plot([0,8],[1,1],'m--','LineWidth',2);title(['Comparison result for rad = ',num2str(j),'0']);
% hold on
% boxplot(bp,g);
% ylim([0 5])
% ylabel(['Normalized unit'])
% set(gcf, 'Color', 'w');
% export_fig(pathsave,'-r300');

%% Radius oriented
% % for i = 2:8
% i = 2;
%     g = [];
%     bp = [];
% 
%     for j = 1:20
%     b = Bfl{j};
%     ba = b(i,:)';
%     [tem,~] = find(ba>500 | ba<0.001);
%     ba(unique(tem))=[];
%     nm = length(ba);
%     bp = [bp;ba];
%     g = [g;repmat(10*j,nm,1)];
%     end
% %% Plot
% folder = ['Result/Bflow/',date,'/'];
% [~,~,~] = mkdir(folder);
% pathsave = [folder,'/Model ',num2str(i),'_',datestr(now,'HHMMSS'),'.png'];
% HM = figure('name',['Model Fitting'],'Position',[100 100 1200 500]);
% figure(HM)
% switch i
%     case 2
%         nama = ['2 Compartments Model Exchange Flow Estimation'];
%     case 3
%         nama = ['2 Compartments Model Uptake Flow Estimation'];
%     case 4
%         nama = ['Maximum Slope Flow Estimation'];
%     case 5
%         nama = ['1 Compartments Model Flow Estimation'];
%     case 7
%         nama = ['2 Compartments Model Exchange Volume Estimation'];
%     case 8
%         nama = ['2 Compartments Model Uptake Volume Estimation'];
%     otherwise
%         nama = ['none'];
% end
% 
% plot([0,20],[1,1],'m--','LineWidth',2);title(['Comparison result for ',nama]);
% hold on
% boxplot(bp,g);
% ylim([0 10])
% ylabel(['Normalized unit'])
% xlabel(['ROI Radius'])
% set(gcf, 'Color', 'w');
% % export_fig(pathsave,'-r300');
% 
% % end