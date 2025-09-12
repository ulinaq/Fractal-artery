% function Eval = Flowval(prm,Grid,node,vnode,segment,vsegment,Connect,x,coSave,radE)
% 2C uptake model analysis 

Co = node.n + vnode.n; Cv = Co+segment.n; Cq = Cv+vsegment.n;
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz); % create meshgrid
fitmodel = 1;
lim = round(lm-1);
Amax = full(max(coSave(:)));
Amin = full(min(coSave(:)));
% coS = 0.1*(2^16)/(Amax-Amin)*(coSave(:,1:lim));
coS = (coSave(:,1:lim));
maXc = max(coS,[],2);
maXc = reshape(maXc,Grid.Nx,Grid.Ny);
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
[~,~,~] = mkdir(folder);

air = 1.7857*full(def_AIF(coS(:,1:lim),segment))'; maxA = max(air);
% air = full(def_AIF(coS(:,1:lim),segment))'; maxA = max(air);
Time = time(1:lim)';
% for Reps = 20:10:200
Reps = 60;


for nm = 1:25
% def circle ROI
Roi = CircleROI(xcell,ycell,Reps,Grid); 
Flreal(nm) = Floreal(prm,Grid,node,vnode,segment,vsegment,Connect,x,Roi);
roi = full(mean(coS(Roi,1:lim)))'; maxR = max(roi);
[Pfit,con, resnorm] = funfit(Time, air, roi, fitmodel);
Flmeas(nm) = Pfit.BF;
Resnorm(nm) = resnorm;
Flerror(nm) = 100*(Flmeas(nm)-Flreal(nm))/Flreal(nm);
if abs(Resnorm(nm))<0.030
pathsave = [folder,'/Fitting ',num2str(nm),' (rad=',num2str(Reps),').png'];
msg = ['Saving ' pathsave]; disp(msg);
figure('name',['Fitting nr ',num2str(nm),' in randomly located circle (rad=',num2str(Reps),')'],'Position',[100 300 1200 300]);
subplot(1,4,1)
imagesc(maXc,[0 1]);colormap(bone);axis image;axis off; title(['ROI (rad=',num2str(Reps),')']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)

subplot(1,4,[2,3,4])
plot(Time,roi,Time,con); title(['Signal fitting (Residual norm = ',num2str(resnorm),', Flow estimation error = ',num2str(round(Flerror(nm))),'%)']);
xlabel('Time[s]')
ylabel('Concentration')
legend('ROI signal','signal fit');
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
% end
% for i = 1:3
%     airn = air + 0.02*i*maxA*randn(size(air));
%     roin = roi + 0.02*i*maxR*randn(size(roi));
%     [Pfit,conc, resnorm] = funfit(Time, airn, roin, 'uptake');
%     Fln(nm,i) = Pfit.BF;
% pathsave = [folder,'/Fitting ',num2str(nm),'with ',num2str(2*i),'% noise (rad=',num2str(Reps),').png'];
% msg = ['Saving ' pathsave]; disp(msg);
% figure('name',[num2str(2*i),'% Noise added (rad=',num2str(Reps),')'],'Position',[100 300 1500 800]);
% subplot(2,2,[1,3])
% imagesc(maXc,[0 100]);colormap(bone);axis image;axis off; title(['ROI (rad=',num2str(Reps),')']);daspect([1 1 1])
% hold on
% imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
% 
% subplot(2,2,2)
% plot(Time,roin,Time,conc); title(['Signal fitting (Residual norm = ',num2str(resnorm),')']);
% xlabel('Time[s]')
% ylabel('Concentration')
% legend('ROI signal','signal fit');
% 
% subplot(2,2,4)
% sz = 20;
% scatter(Flreal(nm),Fln(nm,i),sz,'LineWidth',2); 
% xlabel('Flow model [ml/100ml/min]')
% ylabel([fitmodel,'plasma flow [ml/100ml/min]'])

% set(gcf, 'Color', 'w');
% export_fig(pathsave,'-r300');
% 
end
end

pathsave = [folder,'/Linear fit of Data ', '(rad=',num2str(Reps),').png'];
msg = ['Saving ' pathsave];
disp(msg);
% [p,S] = polyfit(Flreal,Flmeas,1);
% [y_fit,delta] = polyval(p,Flreal,S);

figure('name','Summary')
plot(Flreal,Flmeas,'bo','LineWidth',2);
hold on
plot(0:max(Flmeas),0:max(Flmeas),'r-')
legend('Data','Identity line')
title(['Linear Fit of Data (ROI rad=',num2str(Reps),')']);
xlabel('Flow model [ml/100ml/min]')
ylabel(['plasma flow [ml/100ml/min]'])
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');


% pathsave = [folder,'/Comparison Summary with noise ', '(rad=',num2str(Reps),').png'];
% msg = ['Saving ' pathsave];
% disp(msg);
% legen = {'Original Data','Linear Fit'};
% sz = 20;
% figure('name','Summary with noise')
% plot(Flreal,Flmeas,'bo','LineWidth',2);
% hold on
% plot(Flreal,y_fit,'r-')
% title(['Plasma flow comparison in randomly located circle (rad=',num2str(Reps),')']);
% xlabel('Flow model [ml/100ml/min]')
% ylabel([fitmodel,'plasma flow [ml/100ml/min]'])

% for i = 1:3
%     [p,S] = polyfit(Flreal,Fln(:,i),1);
%     [y_fit,delta] = polyval(p,Flreal,S);
%     legen{end+1} = [num2str(2*i),'% noise data'];
%     legen{end+1} = [num2str(2*i),'% noise Linear Fit'];
%     dsa = [0.2*i (1-0.1*i) (0.5+0.1*i)];
%     plot(Flreal,Fln(:,i),'+','Color',dsa);
%     plot(Flreal,y_fit,'-','Color',0.1+dsa);
% end
% legend(legen);
% set(gcf, 'Color', 'w');
% export_fig(pathsave,'-r300');
pathsave = [folder,'/Data ',num2str(nm),'(rad=',num2str(Reps),').mat'];
Plsave=[Flreal;Flmeas;Flerror];
save(pathsave,'Plsave');
% end
% Flreal = Floreal(prm,Grid,node,vnode,segment,vsegment,Connect,x,Grid.B);

% Evaluate real plasma flow in ROI
function Fl = Floreal(prm,Grid,node,vnode,segment,vsegment,Connect,x,Roi)
Co = node.n + vnode.n; Cv = Co+segment.n; Cq = Cv+vsegment.n;
Skel = zeros(Grid.Nx,Grid.Ny);        
nm = 1;        
for i=1:segment.n
    ns = length(segment.Cnum{i});
    for j = 1:ns
    Skel(segment.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    Skel(vsegment.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end

for i=1:Connect.n
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    Skel(Connect.ind{i}(j)) = nm;
    end
    nm = nm + 1; 
end
for i=1:node.n
    Skel(node.ind{i}) = nm;
    nm = nm + 1; 
end
for i=1:vnode.n
    Skel(vnode.ind{i})= nm;
    nm = nm + 1; 
end

%% Compute the total flow through vessels into the ROI
area = Skel(Roi);
area = unique(nonzeros(area));

sgde = area(area<=segment.n);
vgde = area(area>segment.n & area<=segment.n+vsegment.n) - segment.n;
cgde = area(area>segment.n+vsegment.n & area<=segment.n+vsegment.n+Connect.n) - (segment.n+vsegment.n);
nde = area(area>segment.n+vsegment.n+Connect.n & area<=node.n+segment.n+vsegment.n+Connect.n) - (segment.n+vsegment.n+Connect.n);
vde = area(area>node.n+segment.n+vsegment.n+Connect.n) - (node.n+segment.n+vsegment.n+Connect.n);

% remove unwanted terminal connection
if ~isempty(cgde)
    conA = Connect.AV(cgde,1);
    conV = Connect.AV(cgde,2);
    tmp = cgde;
    for i = 1 : length(cgde)
        idx = ismember(nde,conA(i));
        idy = ismember(vde,conV(i));
%         if any(idx) && x(Cq+cgde(i))<0 || any(idy) && x(Cq+cgde(i))>0
%             cgde;
%         else 
        if any(idx) && any(idy)
            tmp(tmp==cgde(i)) = [];
        elseif any(idx) && x(Cq+cgde(i))>0
            tmp(tmp==cgde(i)) = [];
        elseif any(idy) && x(Cq+cgde(i))<0
            tmp(tmp==cgde(i)) = [];
        end
    end
    cgde = tmp;
end

% remove unwanted vein segment
if ~isempty(nde)
    clog = node.clog(nde);
    clog = nde(clog);
    for i = 1 : length(clog)    % remove clogged vessels
        ab = segment.num(segment.end==clog(i));
        ba = segment.num(segment.init==clog(i));
        idx = ismember(sgde,[ba,ab]);
        sgde(idx)=[];
    end
    junc = node.isterminal(nde);
    nde = nde(~junc);
    for i = 1 : length(nde)     % remove connected and junction vessel
        ba = segment.num(segment.init==nde(i));
        ab = segment.num(segment.end==nde(i));
        % add some missing segments around node
        sgde = unique([sgde;ab';ba']);
    end
    for i = 1 : length(nde)     % remove connected and junction vessel
        ba = segment.num(segment.init==nde(i));
        ab = segment.num(segment.end==nde(i));
        if x(Co+ab)>0   % remove unwanted segment
            ba = ba(x(Co+ba)>0);
            idx = ismember(sgde,ba);
            sgde(idx)=[];
        else
            ba = ba(x(Co+ba)>0);
            idx = ismember(sgde,[ba,ab]);
            sgde(idx)=[];
        end
    end
end

% remove unwanted venous segment
if ~isempty(vde)
    clog = vnode.clog(vde);
    clog = vde(clog);
    for i = 1 : length(clog)    % remove clogged vessel
        ab = segment.num(segment.end==clog(i));
        ba = segment.num(segment.init==clog(i));
        idx = ismember(vgde,[ba,ab]);
        vgde(idx)=[];
    end
    junc = vnode.isterminal(vde);
    tde = vde(~junc);
    for i = 1 : length(tde)     % remove connected and junction vessel
        ba = vsegment.num(vsegment.init==tde(i));
        ab = vsegment.num(vsegment.end==tde(i));
        % add some missing segments around node
        vgde = unique([vgde;ab';ba']);
    end
    for i = 1 : length(tde)     % remove connected and junction vessel
        ba = vsegment.num(vsegment.init==tde(i));
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)<0
            ba = ba(x(Cv+ba)<0);
            idx = ismember(vgde,ba);
            vgde(idx)=[];
        else
            ba = ba(x(Cv+ba)<0);
            idx = ismember(vgde,[ba,ab]);
            vgde(idx)=[];
        end
    end
    junc = vnode.isterminal(vde);
    tde = vde(junc);
    for i = 1 : length(tde)
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)>0
            idx = ismember(vgde,ab);
            vgde(idx)=[];        
        end
    end
end

% Total flow into ROI
Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*60/100/Grid.volume;
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*60/100/(prm.pixel^2)/0.55;
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*60/10/prm.pixel;
% Fl = sum(abs(x([Co+sgde;Cq+cgde])))*0.55*600/100/prm.pixel;
end

function [Vol,Flo] = Comp(Grid,node,vnode,segment,vsegment,x)
cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny);
fA = sparse(Grid.Nx,Grid.Ny); fV = sparse(Grid.Nx,Grid.Ny);
Co = node.n + vnode.n;  Cv = Co + segment.n;
for i=1:segment.n
    ns = length(segment.Cnum{i});
    fA(node.avind{segment.end(i)})=x(Co+i);
    for j = 1:ns
    cA(segment.distrind{i}{j})=1;
    fA(segment.distrind{i}{j})=x(Co+i);
    end
end
for i=1:vsegment.n
    ns = length(vsegment.Cnum{i});
    fV(vnode.avind{vsegment.end(i)})=x(Cv+i);
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=1;
    fV(vsegment.distrind{i}{j})=x(Cv+i);
    end
end
for i=1:node.n
    cA(node.avind(i))=1;
end
for i=1:vnode.n
    cV(vnode.avind(i))=1;
end
Vol = (cA+cV)*0.55;
Vol(~Grid.B) = 0;
Flo = (fA+fV)*0.55*60/100/Grid.volume;
Flo(~Grid.B) = 0;
end

% creat a random located circle ROI with a predetermined radius
    function area = CircleROI(xcell,ycell,Rep,Grid)
        Reps = 1*Rep;
        Rx = randi([Reps Grid.Nx-Reps]);
        Ry = randi([Reps Grid.Ny-Reps]);
        Center = sub2ind([Grid.Nx Grid.Ny],Rx,Ry);
        area=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Rep^2;
    end

% Define AIF region as some first few segments from root segment
    function AIF = def_AIF(c,segment)
    [a,b]=maxk(segment.rad,6);
    reg = segment.distrind(b);
    as=[];
    for i = 1:length(reg)
        as = [as;cell2mat(reg{i})];
    end
    reg = unique(as);
    AIF = mean(c(reg,:));    
    end