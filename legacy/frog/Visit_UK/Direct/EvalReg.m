% function Eval = Flowval(prm,Grid,node,vnode,segment,vsegment,Connect,x,coSave,radE)
Co = node.n + vnode.n; Cv = Co+segment.n; Cq = Cv+vsegment.n;
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz); % create meshgrid
lim = round(lm-1);
Amax = full(max(coSave(:)));
Amin = full(min(coSave(:)));
% coS = 0.1*(2^16)/(Amax-Amin)*(coSave(:,1:lim));
coS = coSave(:,1:lim)*1000;
% coS=int16(1000*full(coSave(:,1:lim)));
maXc = reshape(max(coS,[],2),Grid.Nx,Grid.Ny);
folder = ['Result/EvalRad/',date,'-2/'];
[~,~,~] = mkdir(folder);

% air = 1.7857*full(def_AIF(coS(:,1:lim),segment))'; maxA = max(air);
air = full(def_AIF(coS(:,1:lim),segment,Grid))'; maxA = max(air);
Time = time(1:lim);
tic;
Rep = 100;
Reps = 1*Rep;
Mn = 200;
Bflow = zeros(8,Mn);
rxp = randperm(Grid.Nx-2*Reps);
ryp = randperm(Grid.Ny-2*Reps);
for jk = 1:Mn
Rx = Reps + rxp(mod(jk,Mn)+randi(3));
Ry = Reps + ryp(mod(jk,Mn-30)+randi(30));
Center = sub2ind([Grid.Nx Grid.Ny],Rx,Ry);
Roi=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Reps^2;
% Roi = xcell > 1;
% Roi = logical(Froi{jk});
% def circle ROI
    [Vol,Flo] = Comp(Grid,node,vnode,segment,vsegment,Connect,x);
    BVreal = mean(Vol(Roi))*100*0.55;
    Flreal = Floreal(prm,Grid,node,vnode,segment,vsegment,Connect,x,Roi)*0.55;
    if Flreal~=0
    roi = full(mean(coS(Roi,1:lim)))'; maxR = max(roi);
    [Pfit1,con1, resnorm1] = funfit(Time, air, roi, 2,Flreal,BVreal); % 2CM exchange model
    Flmeas(1) = Pfit1.BF;
    [Pfit2,con2, resnorm2] = funfit(Time, air, roi, 1,Flreal,BVreal); % 2CM uptake model
    Flmeas(2) = Pfit2.BF;
    [Pfit3,con3, resnorm3] = funfit(Time, air, roi, 3,Flreal,BVreal); % Maximum Slope model
    Flmeas(3) = Pfit3.BF;
    [Pfit4,con4, resnorm4] = funfit(Time, air, roi, 4,Flreal,BVreal); % Compartment model
    Flmeas(4) = Pfit4.BF;
    Flerror = 100*(Flmeas-Flreal)/Flreal;
    Bflow(:,jk) = [Flreal,Pfit1.BF,Pfit2.BF,Pfit3.BF,Pfit4.BF,BVreal,Pfit1.BV,Pfit2.BV];
% Create a table
    Tbl.label = {'2CM Exchange';'2CM Uptake';'Maximum Slope';'1-Compartment';'Real'};
    Tbl.CME = [Pfit1.BF;Pfit1.BV;Pfit1.BMTT;Pfit1.PS;Pfit1.IV;Pfit1.IMTT;Pfit1.Ktrans;Pfit1.Kep;Pfit1.EF;Pfit1.EV;Pfit1.EMTT];
    Tbl.CMU = [Pfit2.BF;Pfit2.BV;Pfit2.BMTT;Pfit2.PS;NaN;NaN;Pfit2.Ktrans;NaN;Pfit2.EF;NaN;NaN];
    Tbl.MS = [Pfit3.BF;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN];
    Tbl.OCM = [Pfit4.BF;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;Pfit4.EV;Pfit4.EMTT];
    Tbl.Real = [Flreal;BVreal;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN];
    Tbl.VarNames = {'Plasma Flow [ml/100ml/min]','Plasma Volume [ml/100ml]','Plasma MTT [sec]','Permeability-surface [ml/100ml/min]',...
        'Interstitial Volume [ml/100ml]','Interstitial MTT [sec]','Ktrans [ml/100ml/min]','kep [ml/100ml/min]','Extraction Fraction [%]','Extracellular Volume [%]','Extracellular MTT [sec]'};
    T = table(Tbl.CME,Tbl.CMU,Tbl.MS,Tbl.OCM,Tbl.Real,'RowNames',Tbl.VarNames);
    end
% Plot figure
if abs(Bflow(1,jk)-Bflow(2,jk))<0.2*Bflow(1,jk)
H = figure('name',['Plasma flow comparison in randomly located circle (rad=',num2str(Reps),')'],'Position',[100 100 1600 800]);
    pathsave = [folder,'/Reg',datestr(now,'ddmmyy_HHMM'),'.png'];
    msg = ['Saving ' pathsave]; disp(msg);
figure(H)
clf;
subplot(2,3,1)
imagesc(maXc,[0 max(maXc(:))/10000]);colormap(bone);axis image;axis off; title(['ROI (rad=',num2str(Reps),')']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
% plot(GV,'c','YData',vnode.XData/prm.pixel,'XData',vnode.YData/prm.pixel,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% plot(GA,'r','YData',node.XData/prm.pixel,'XData',node.YData/prm.pixel,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1]);

subplot(2,3,[2,3])
plot(Time,roi,'k-',Time,con3,'r:','LineWidth',1); title(['Signal fitting comparison']);
hold on
plot(Time,con4,'b--',Time,con2,'g-.',Time,con1,'m-','LineWidth',1);
xlim([0 60]); xlabel('Time[s]');
ylabel('Concentration[mM]');
legend('ROI signal','MS fitting','1CM fitting','2CM Uptake fitting','2CM Exchange fitting');
set(gcf, 'Color', 'w');

subplot(2,3,6)
sz = 20;
amin = min([ones(4,1)*Flreal;Flmeas(:)]); amax = max([ones(4,1)*Flreal;Flmeas(:)]);
plot([0,amax],[0,amax],'k','LineWidth',1);
hold on
plot(Flreal,Flmeas(3),'r*','LineWidth',2);
plot(Flreal,Flmeas(4),'b*','LineWidth',2);
plot(Flreal,Flmeas(2),'g*','LineWidth',2);
plot(Flreal,Flmeas(1),'m*','LineWidth',2);
xlabel('Flow model [ml/100ml/min]')
ylabel(['Prediction plasma flow [ml/100ml/min]'])
legend({'Identity line','MS fitting','OCM fitting','2CM Uptake fitting','2CM Exchange fitting'},'Location','northwest');

% subplot(2,2,3)
uitable('Data',T{:,:},'ColumnName',Tbl.label,'RowName',T.Properties.RowNames,'Position',[220 150 800 240]);
set(gcf, 'Color', 'w');
% export_fig(pathsave,'-r300');
end
end
Bflow100=Bflow;
% Bflow170=[Bflow170,Bflow];
% b =zeros(size(Bflow));
% for i=1:length(Bflow)
% b(1:5,i)=Bflow(1:5,i)/Bflow(1,i);
% b(6:8,i)=Bflow(6:8,i)/Bflow(6,i);
% end
% [~,tem] = find(b>200 | b<0.01);
% b(:,unique(tem))=[];
% Flmean = mean(b,2);
% Flmean(:,2) = std(b,0,2)
toc

% Evaluate real plasma flow in ROI
function Fl = Floreal(prm,Grid,node,vnode,segment,vsegment,Connect,x,Roi,ROI)
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
% Fl = sum(abs([x(Co+sgde)./(pi*segment.rad(sgde).^2);x(Cv+vgde)./(pi*vsegment.rad(vgde).^2);x(Cq+cgde)./(pi*Connect.rad(cgde).^2)]))*0.55*60/100;
Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*6000/(nnz(Roi)*Grid.volume);
% Fl = sum(abs(x([Co+sgde;Cv+vgde;Cq+cgde])))*0.55*60/10/prm.pixel;
end

function [Vol,Flo] = Comp(Grid,node,vnode,segment,vsegment,Connect,x)
cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny); cO = sparse(Grid.Nx,Grid.Ny);
fA = sparse(Grid.Nx,Grid.Ny); fV = sparse(Grid.Nx,Grid.Ny); Vol = zeros(Grid.Nx,Grid.Ny);
Co = node.n + vnode.n;  Cv = Co + segment.n;
for i=1:segment.n
    Vols = (pi*segment.rad(i)^2*Grid.hx);
    ns = length(segment.Cnum{i});
    for j = 1:ns
    cA(segment.distrind{i}{j})=Vols/numel(segment.distrind{i}{j});
    end
end
for i=1:vsegment.n
    Vols = (pi*vsegment.rad(i)^2*Grid.hx);
    ns = length(vsegment.Cnum{i});
    for j = 1:ns
    cV(vsegment.distrind{i}{j})=Vols/numel(vsegment.distrind{i}{j});
    end
end
for i=1:node.n
    cA(node.ind{i})=Grid.volume/numel(node.ind{i});
end
for i=1:vnode.n
    cV(vnode.ind{i})=Grid.volume/numel(vnode.ind{i});
end
for i=1:Connect.n
    Vols = (pi*Connect.rad(i)^2*Grid.hx);
    ns = length(Connect.Cnum{i});
    for j = 1:ns
    cO(Connect.distrind{i}{j})=Vols/numel(Connect.distrind{i}{j});
    end
end
Vol = (cA+cV+cO)/Grid.volume;
Vol(~Grid.B) = 0;
Flo = (fA+fV)*0.55*6000/Grid.volume;
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
function AIF = def_AIF(c,segment,Grid)
    [a,b]=maxk(segment.rad,4);
    k = 1;
    for i = 1:length(b)
    Vols = (pi*segment.rad(b(i))^2*Grid.hx);
    ns = length(segment.Cnum{b(i)});
    for j = 1:ns
    cV(k,:)=sum(c(segment.distrind{b(i)}{j},:),1)*Grid.volume/Vols;
    k = k+1;
    end
    end
     AIF = mean(mean(cV,1),1);
end