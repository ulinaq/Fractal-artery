% function [N] = integralC(Grid,node,segment,vnode,vsegment,Connect,coSave,x,Time,ROI)
Co = node.n + vnode.n;  Cv = Co + segment.n;    Cq = Cv+vsegment.n; 
% ROI choosen 
gambar = [1:11];
% for nm = 5:6
% nois = [1,2,4,6,8,10];
% nois = nois(nm);
Vol = Comp(Grid,node,vnode,segment,vsegment,Connect);
for jk =1:11
% jk=6;
Roi = logical(Froi{jk});
% bw perimeter outside ROI
Rper = bwmorph(Roi,'thin',1);
% Skeleton mapping segment
Skel = SkelAll(Grid,node,vnode,segment,vsegment,Connect);
% Concentration array time series
lim = size(coSave,2)-1;
coS = coSave(:,1:lim);
maXc = full(reshape(max(coS,[],2),Grid.Nx,Grid.Ny));
roi = full(mean(coS(Roi,1:lim)));
% Find segment and ROI intersection 
[indA,sIn,indV,sOut,indC] = Sections(node,vnode,segment,vsegment,Connect,Skel,Rper,x);
Fri = x(Co+sIn.A);
Fro = x(Cv+sOut.V);
c=[];   cv=[];
for i = 1:length(Fri)
    c(i,:) = mean(cISave(Co+NCs+segment.Cnum{sIn.A(i)}(sIn.indA{i}),1:lim),1); 
end
if ~isempty(sIn.V)
for i = length(sIn.A)+1:length(sIn.A)+length(sIn.V)
    c(i,:) = mean(cISave(Co+NCs+NAs+vsegment.Cnum{sIn.V(i)}(sIn.indV{i}),1:lim),1); 
end
end 
if ~isempty(sIn.C)
for i = length(sIn.A)+length(sIn.V)+1:length(sIn.A)+length(sIn.V)+length(sIn.C)
    c(i,:) = mean(cISave(node.n+Connect.Cnum{sIn.C(i)}(sIn.indC{i}),1:lim),1); 
end
end 
for i = 1:length(Fro)
    cv(i,:) = mean(cISave(Co+NCs+NAs+vsegment.Cnum{sOut.V(i)}(sOut.indV{i}),1:lim),1);
end
% c = c + 0.01*nois*max(c(:))*randn(size(c));
% cv = cv + 0.01*nois*max(cv(:))*randn(size(cv));
% roi = roi + 0.01*nois*max(roi(:))*randn(size(roi));

t = time(1:lim);
Cai = INT_TRAP(t,Fri'*c);
Cvo = INT_TRAP(t,Fro'*cv);
Cr = INT_TRAP(t,roi);
N = roi(1:lim)*(sum(Roi(:)))*Grid.volume;%sum(cVol(Roi));%
N2 = INT_TRAP(t,Fri'*c-Fro'*cv);
Cr = trapz(t,N);
Cr2 = trapz(t,N2);
[Cr(end) Cr2(end)];
(Cr(end)-Cr2(end))/Cr2(end)*100;
% 
%% Evaluate blood flow
    BVreal = mean(Vol(Roi))*100;
    Flreal = sum(Fri)*0.55*6000/(nnz(Roi)*Grid.volume);
    if Flreal~=0
    maxR = max(roi);
    [Pfit1,con1, resnorm1] = funfit(t, (Fri'*c/sum(Fri))', roi',2);% 2CM X model
    [Pfit2,con2, resnorm2] = funfit(t, (Fri'*c/sum(Fri))', roi', 1);% 2CM uptake model
    [Pfit3,con3, resnorm3] = funfit(t, (Fri'*c/sum(Fri))', roi', 3);% Maximum Slope model
    [Pfit4,con4, resnorm4] = funfit(t, (Fri'*c/sum(Fri))', roi', 4);% Compartment model
    Flmeas(1) = Pfit1.BF;
    Flmeas(2) = Pfit2.BF;
    Flmeas(3) = Pfit3.BF;
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
folder = ['Result/EvalROI2/',date,'/'];
[~,~,~] = mkdir(folder);
pathsave = [folder,'/Reg',datestr(now,'ddmmyy_HHMMSS'),'.png'];
H = figure('name',['Plasma flow comparison in ROI ',num2str(jk)],'Position',[100 100 1600 800]);
    pathsave = [folder,'/Reg',datestr(now,'ddmmyy_HHMM'),'.png'];
    msg = ['Saving ' pathsave]; disp(msg);
figure(H)
clf;
subplot(2,3,1)
imagesc(maXc,[0 max(maXc(:))/10000]);colormap(bone);axis image;axis off; daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
% plot(GV,'c','YData',vnode.XData/prm.pixel,'XData',vnode.YData/prm.pixel,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% plot(GA,'r','YData',node.XData/prm.pixel,'XData',node.YData/prm.pixel,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1]);

subplot(2,3,[2,3])
plot(t,roi,'k-',t,con3,'r:','LineWidth',1); title(['Signal fitting comparison']);
hold on
plot(t,con4,'b--',t,con2,'g-.',t,con1,'m-','LineWidth',1);
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

subplot(2,3,[4,5])
uitable('Data',T{:,:},'ColumnName',Tbl.label,'RowName',T.Properties.RowNames,'Position',[220 150 800 240]);
set(gcf, 'Color', 'w');

end

%% Functions
% Locate the input and output vessels in ROI
function [indA,sIn,indV,sOut,indC] = Sections(node,vnode,segment,vsegment,Connect,Skel,Roi,x)
Co = node.n + vnode.n;  Cv = Co + segment.n;    Cq = Cv+vsegment.n; 
Rper = bwmorph(Roi,'remove');
Rper = imdilate(Rper, strel('disk',2));
indA = find(Rper&segment.skel);
indV = find(Rper&vsegment.skel);
indC = find(Rper&logical(Skel)&~segment.skel&~vsegment.skel);
area = Skel(Roi);
area = unique(nonzeros(area));

% Separate segments and nodes inside ROI
sgde = area(area<=segment.n);
vgde = area(area>segment.n & area<=segment.n+vsegment.n) - segment.n;
cgde = area(area>segment.n+vsegment.n & area<=segment.n+vsegment.n+Connect.n) - (segment.n+vsegment.n);
nde = area(area>segment.n+vsegment.n+Connect.n & area<=node.n+segment.n+vsegment.n+Connect.n) - (segment.n+vsegment.n+Connect.n);
vde = area(area>node.n+segment.n+vsegment.n+Connect.n) - (node.n+segment.n+vsegment.n+Connect.n);

% Find the vascular network segment that provide blood into the ROI
% remove unwanted terminal connection
if ~isempty(cgde)
    conA = Connect.AV(cgde,1);
    conV = Connect.AV(cgde,2);
    tmp = cgde;
    for i = 1 : length(cgde)
        idx = ismember(nde,conA(i));
        idy = ismember(vde,conV(i));
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

% remove unwanted artery segment
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
    % add some missing segments around node
    for i = 1 : length(nde)     
        ba = segment.num(segment.init==nde(i));
        ab = segment.num(segment.end==nde(i));
        sgde = unique([sgde;ab';ba']);
    end
    % remove connected and junction vessel
    for i = 1 : length(nde)     
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
        if x(Cv+ab)<0 % flow from parent to child
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
% remove no flow vessels
sgde(abs(x(Co+sgde))<1e-13) = [];
vgde(abs(x(Cv+vgde))<1e-13) = [];
cgde(abs(x(Cq+cgde))<1e-13) = [];
sIn.A = sgde;  sIn.V = vgde;  sIn.C = cgde;
% Segment intersect with ROI
if ~isempty(sgde)
    for i = 1:length(sgde)
        sn = sgde(i);
        indeks = false(length(segment.Cnum{sn}),1);
        for j = 1:length(segment.Cnum{sn})
            indeks(j) = any(intersect(segment.distrind{sn}{j},indA));
        end
        sIn.indA{i} = find(indeks);
    end
end
if ~isempty(vgde)
    for i = 1:length(vgde)
        sn = vgde(i);
        indeks = false(length(vsegment.Cnum{sn}),1);
        for j = 1:length(vsegment.Cnum{sn})
            indeks(j) = any(intersect(vsegment.distrind{sn}{j},indV));
        end
        sIn.indV{i} = find(indeks);
    end
end
if ~isempty(cgde)
    for i = 1:length(cgde)
        sn = cgde(i);
        indeks = false(length(Connect.Cnum{sn}),1);
        for j = 1:length(Connect.Cnum{sn})
            indeks(j) = any(intersect(Connect.distrind{sn}{j},indC));
        end
        sIn.indC{i} = find(indeks);
    end
end

%% %%%%%%%%%%%%
% Find the outward flow segment
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
        if any(idx) && any(idy)
            tmp(tmp==cgde(i)) = [];
        elseif any(idx) && x(Cq+cgde(i))<0
            tmp(tmp==cgde(i)) = [];
        elseif any(idy) && x(Cq+cgde(i))>0
            tmp(tmp==cgde(i)) = [];
        end
    end
    cgde = tmp;
end

% remove unwanted artery segment
if ~isempty(nde)
    clog = node.clog(nde);
    clog = nde(clog);
    for i = 1 : length(clog)    % remove clogged vessels
        ab = segment.num(segment.end==clog(i));
        ba = segment.num(segment.init==clog(i));
        idx = ismember(sgde,[ba,ab]);
        sgde(idx)=[];
    end
    % add some missing segments around junction node
    junc = node.isterminal(nde);
    tde = nde(~junc);
    for i = 1 : length(tde)     
        ba = segment.num(segment.init==tde(i));
        ab = segment.num(segment.end==tde(i));
        % add some missing segments around node
        sgde = unique([sgde;ab';ba']);
    end
    for i = 1 : length(tde)     % remove connected and junction vessel
        ba = segment.num(segment.init==tde(i));
        ab = segment.num(segment.end==tde(i));
        if x(Co+ab)<0   % remove unwanted segment
            ba = ba(x(Co+ba)<0);
            idx = ismember(sgde,ba);
            sgde(idx)=[];
        else
            ba = ba(x(Co+ba)<0);
            idx = ismember(sgde,[ba,ab]);
            sgde(idx)=[];
        end
    end
    junc = node.isterminal(nde);
    tde = nde(junc);
    for i = 1 : length(tde)
        ab = segment.num(segment.end==tde(i));
        if x(Co+ab)>0
            idx = ismember(sgde,ab);
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
        if x(Cv+ab)>0 % flow from parent to child
            ba = ba(x(Cv+ba)>0);
            idx = ismember(vgde,ba);
            vgde(idx)=[];
        else 
            ba = ba(x(Cv+ba)>0);
            idx = ismember(vgde,[ba,ab]);
            vgde(idx)=[];
        end
    end
    junc = vnode.isterminal(vde);
    tde = vde(junc);
    for i = 1 : length(tde)
        ab = vsegment.num(vsegment.end==tde(i));
        if x(Cv+ab)<0
            idx = ismember(vgde,ab);
            vgde(idx)=[];        
        end
    end
end
% remove no flow vessels
sgde(abs(x(Co+sgde))<1e-12) = [];
vgde(abs(x(Cv+vgde))<1e-12) = [];
cgde(abs(x(Cq+cgde))<1e-12) = [];
sOut.A = sgde;  sOut.V = vgde;  sOut.C = cgde;
% Segment intersect with ROI
if ~isempty(sgde)
    for i = 1:length(sgde)
        sn = sgde(i);
        indeks = false(length(segment.Cnum{sn}),1);
        for j = 1:length(segment.Cnum{sn})
            indeks(j) = any(intersect(segment.distrind{sn}{j},indA));
        end
        sOut.indA{i} = find(indeks);
    end
end
if ~isempty(vgde)
    for i = 1:length(vgde)
        sn = vgde(i);
        indeks = false(length(vsegment.Cnum{sn}),1);
        for j = 1:length(vsegment.Cnum{sn})
            indeks(j) = any(intersect(vsegment.distrind{sn}{j},indV));
        end
        sOut.indV{i} = find(indeks);
    end
end
if ~isempty(cgde)
    for i = 1:length(cgde)
        sn = cgde(i);
        indeks = false(length(Connect.Cnum{sn}),1);
        for j = 1:length(Connect.Cnum{sn})
            indeks(j) = any(intersect(Connect.distrind{sn}{j},indC));
        end
        sOut.indC{i} = find(indeks);
    end
end

end
% Create skeleton number of vascula element
function [Skel] = SkelAll(Grid,node,vnode,segment,vsegment,Connect)
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
end
% trapesium integral cumulative
function trapz_int_final = INT_TRAP(t,c)
    n = length(t);
    nC = size(c,1);
    dt = repmat(t(2:n)-t(1:n-1),1,nC)';
    trapz_int = dt.*(c(:,1:n-1)+c(:,2:n))/2;
    trapz_int_2 = [zeros(nC,1),trapz_int]; 
    trapz_int_final = cumsum(trapz_int_2);
end

function trapz_int_final = INT_T(t,c)
    n = length(t);
    dt = t(2:n)-t(1:n-1);
    trapz_int = dt.*(c(1:n-1,:)+c(2:n,:))/2;
    trapz_int_2 = [zeros();trapz_int]; 
    trapz_int_final = cumsum(trapz_int_2);
end

function [bf,bg] = Rekap(BF,BG)
ab=[];
for j=1:6
b=0;
a=0;
for i = 1:11
    if abs(BG{i}(2,j))<300
        b=b+abs(BG{i}(2,j));
        a=a+1;
    end
end
ab(j)=b/a;
end



for j=1:6
bg=[];
for i = 1:11
    bg(:,i)=round(BG{i}(1,:)',2);
end
end
bg(:,end+1) = ab;

ab=[];
for j=1:7
b=0;
a=0;
for i = 1:11
    if abs(BF{i}(2,j))<300
        b=b+abs(BF{i}(2,j));
        a=a+1;
    end
end
ab(j)=b/a;
end

for j=1:7
bf=[];
for i = 1:11
    bf(:,i)=round(BF{i}(1,:)',2);
end
end
bf(:,end+1) = ab;
end

% Compute volume fraction/Porosity
function Vol = Comp(Grid,node,vnode,segment,vsegment,Connect)
cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny); cO = sparse(Grid.Nx,Grid.Ny);
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
    cA(node.ind{i})=Grid.volume;
end
for i=1:vnode.n
    cV(vnode.ind{i})=Grid.volume;
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
end

function AIF = def_AIF(c,segment,Grid,kx)
    [a,b]=maxk(segment.rad,kx);
    k = 1;
    for i = 1:length(b)
    Vols = (pi*segment.rad(b(i))^2*Grid.hx);
    ns = length(segment.Cnum{b(i)});
    for j = round(ns/2):ns
    cV(k,:)=sum(c(segment.distrind{b(i)}{j},:),1)*Grid.volume/Vols;
    k = k+1;
    end
    end
     AIF = mean(mean(cV,1),1);
end
