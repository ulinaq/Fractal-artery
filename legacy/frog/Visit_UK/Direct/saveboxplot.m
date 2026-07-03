Bfl = Bflo;
name = ['Plasma flow estimation']
legendEntries = {'MS' '1CM' '2CM Uptake' '2CM Exchange'};
labels = {'0.6' '1.2' '1.8' '2.4' '3.0' '3.6' '4.2' '4.8' '5.4' '6.0' '6.6' '7.2' '7.8' '8.4' '9.0' '9.6' '10.2' '10.8' '11.4' '12.0'};

bpi=[]; 
DeviceColors = {'r' 'm' 'b' 'g' 'c' 'g' 'm' 'b' };
delta = linspace(-0.3,.3,4); %// define offsets to distinguish plots
width = 0.2
j = 1;

HM = figure('name',['Model Fitting'],'Position',[100 100 1200 600]);
plot([0,22],[1,1],'r--','LineWidth',1, 'HandleVisibility','off' );title(name);
hold on

% for i = 7:8
urut = [4,5,3,2];
for irs = 1:4
    i = urut(irs);
    bp=[]; g=[];
    for jk = 1:20
        lm(jk) = length(Bfl{jk});
    end
    k = min(lm);
    for jk = 1:20
        b = Bfl{jk};
        temp = b(i,:);
        [tem] = find(temp>1000 | temp<0);
        temp(unique(tem))=[];
        temp = (temp-1)*100;
        bp = [bp;temp'];
        g = [g;repmat(jk,length(temp),1)];
        bpi(i,jk) = length(temp);
    end
    
    boxplot(bp,g, 'Whisker', 1.5, 'PlotStyle','compact', 'Symbol','.',...
        'ExtremeMode','clip', 'Colors',DeviceColors{i}, 'position',(1:20)+delta(j),...
        'widths',width, 'labels',labels);
    plot(NaN,1,'color',DeviceColors{i}); %// dummy plot for legend
    j = j+1
end
    xlim([0.4 20+2*delta(end)]) %// adjust x limits, with room for legend
    ylim([-100 500])
    ylabel(['Estimation Error [%]'])
    xlabel(['ROI radius [mm]'])
    legend(legendEntries);
    
legendEntries = {'2CM Uptake' '2CM Exchange'};
name = ['Blood volume estimation'];
delta = linspace(-0.1,.1,2); %// define offsets to distinguish plots
j = 1;

HM = figure('name',['Blood Volume'],'Position',[100 100 1200 600]);
plot([0,22],[1,1],'r--','LineWidth',1, 'HandleVisibility','off' );title(name);
hold on

for i = 8:-1:7

    bp=[]; g=[];
    for jk = 1:20
        lm(jk) = length(Bfl{jk});
    end
    k = min(lm);
    for jk = 1:20
        b = Bfl{jk};
        temp = b(i,:);
        [tem] = find(temp>1000 | temp<0);
        temp(unique(tem))=[];
        temp = (temp-1)*100;
        bp = [bp;temp'];
        g = [g;repmat(jk,length(temp),1)];
        bpi(i,jk) = length(temp);
    end
    
    boxplot(bp,g, 'Whisker', 1.5, 'PlotStyle','compact', 'Symbol','.',...
        'ExtremeMode','clip', 'Colors',DeviceColors{i}, 'position',(1:20)+delta(j),...
        'widths',width, 'labels',labels);
    plot(NaN,1,'color',DeviceColors{i}); %// dummy plot for legend
    j = j+1
end
    xlim([0.4 20+2*delta(end)]) %// adjust x limits, with room for legend
    ylim([-100 300])
    ylabel(['Estimation Error [%]'])
    xlabel(['ROI radius [mm]'])
    legend(legendEntries);