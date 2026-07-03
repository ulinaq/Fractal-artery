fitmodel = 4;
MaxReps = 150;
dC = 2/MaxReps;
switch fitmodel
    case 1
        Tmodel = '2CMU';
    case 2
        Tmodel = '2CME';
    case 3
        Tmodel = 'MS';
    case 4
        Tmodel = 'OCM';
end
folder = ['Result/Fig/',Tmodel,date,'/'];
pathsave = [folder,'/Comparison ',date,Tmodel,'.png'];
msg = ['Saving ' pathsave];
disp(msg);
Maxv = 300;
legen = {'Identity line'};
h1 = figure('name','Summary','Position',[100 300 1400 500]);
subplot(1,2,1)
plot(0:100:Maxv,0:100:Maxv,'k-','LineWidth',1)
hold on
h2 = figure('name','Linear Regression','Position',[100 300 600 500]);
plot(0:100:Maxv,0:100:Maxv,'k-','LineWidth',1)
hold on
% title(['Linear Fit of Data)']);
% xlabel('Flow model [ml/100ml/min]')
% ylabel('2CME Fitting plasma flow [ml/100ml/min]')
CoreR =[];
for Reps = 10:10:MaxReps
% Reps = 40;
load([folder,'Data 50(rad=',num2str(Reps),').mat'])
Flreal = Plsave(1,:);
Flmeas = Plsave(2,:);
%     [a,b] = find(Flmeas>2*Maxv);
%     Flmeas(b) = [];
%     Flreal(b) = [];
%     [a,b] = find(Flreal>Maxv);
%     Flmeas(b) = [];
%     Flreal(b) = [];
R = corrcoef(Flreal,Flmeas);
[p,S] = polyfit(Flreal,Flmeas,1);
[y_fit,delta] = polyval(p,[0,Flreal,Maxv],S);
% figure('name','Summary')
legen{end+1} = ['Rad = ',num2str(Reps)];
dsa = [((dC*Reps)-1+2*dC) (1-4*(dC*Reps-1).^2)  (1+2*dC-dC*Reps)];% (10*(0.0085*Reps-0.5).^2-0.3)]; %
dsa = min(dsa,1);   dsa = max(dsa,0);
figure(h1)
plot(Flreal,Flmeas,'*','Color',dsa);
% hold on
figure(h2)
% plot(0:100:Maxv,0:100:Maxv,'r-','LineWidth',2)
plot([0,Flreal,Maxv],y_fit,'-','LineWidth',1,'Color',dsa)
% title(['Linear Fit of data (rad = ',num2str(Reps),') Corr = ',num2str(min(R(:)))]);
% xlabel('Flow model [ml/100ml/min]')
% ylabel('Fitting plasma flow [ml/100ml/min]')
% legend(['Data rad = ',num2str(Reps)],'Identity line','Linear fit');
CoreR(end+1) = min(R(:));
end
figure(h1)
title([Tmodel,' Comparison Data']);
xlabel('Flow model [ml/100ml/min]')
ylabel([Tmodel,' Fitting plasma flow [ml/100ml/min]'])
axis([0 Maxv 0 4/3*Maxv])
legend(legen)
set(gcf, 'Color', 'w');
% export_fig(pathsave,'-r300');

pathsave = [folder,'/Summary ',Tmodel,'.png'];
subplot(1,2,2)
for Reps = 10:10:MaxReps
dsa = [(dC*Reps-1+2*dC) (1-4*(dC*Reps-1).^2)  (1+2*dC-dC*Reps)];% (10*(0.0085*Reps-0.5).^2-0.3)]; %
dsa = min(dsa,1);   dsa = max(dsa,0);
plot(Reps,CoreR(Reps/10),'*','LineWidth',2,'Color',dsa)
hold on
end
title([Tmodel,' Correlation coefficient']);
xlabel('ROI radius')
ylabel('Correlation')
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');

pathsave = [folder,'/Linear Regression ',Tmodel,'.png'];
figure(h2)
title([Tmodel,' Linear Regression of Data']);
xlabel('Flow model [ml/100ml/min]')
ylabel([Tmodel,' Fitting plasma flow [ml/100ml/min]'])
axis([0 Maxv 0 4/3*Maxv])
legend(legen)
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
