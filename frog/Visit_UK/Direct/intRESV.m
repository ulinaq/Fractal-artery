% function [N] = integralC(Grid,node,segment,vnode,vsegment,Connect,coSave,x,Time,ROI)
Co = node.n + vnode.n;  Cv = Co + segment.n;    Cq = Cv+vsegment.n; 
% ROI choosen 
gambar = [1,2,6,7,9,11];
for jk =1:2
Roi = logical(Froi{jk});
% bw perimeter outside ROI
Rper = bwmorph(Roi,'thin',1);
% Skeleton mapping segment
Skel = SkelAll(Grid,node,vnode,segment,vsegment,Connect);
% Concentration array time series
lim = size(coSave,2)-1;
for nm = 1:2
nois = [1,2,4,6,8,10];
nois = nois(nm);
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
for i = 1:length(Fro)
    cv(i,:) = mean(cISave(Co+NCs+NAs+vsegment.Cnum{sOut.V(i)}(sOut.indV{i}),1:lim),1);
end
c = c + 0.001*nois*max(c(:))*randn(size(c));
cv = cv + 0.001*nois*max(cv(:))*randn(size(cv));
roi = roi + 0.001*nois*max(roi(:))*randn(size(roi));

t = time(1:lim);
Cai = INT_TRAP(t,Fri'*c);
Cvo = INT_TRAP(t,Fro'*cv);
Cr = INT_TRAP(t,roi);
N = roi(1:lim)*(sum(Roi(:)))*Grid.volume;%sum(cVol(Roi));%
N2 = INT_TRAP(t,Fri'*c-Fro'*cv);
Cr = trapz(t,N);

Cr2 = trapz(t,N2);
[Cr(end) Cr2(end)];
(Cr(end)-Cr2(end))/Cr2(end)*100

%% Evaluate blood flow
F = sum(Fri);
Fr = [F,max(diff(N)./diff(t'))/max(Fri'*c/sum(Fri)),max((diff(N)./diff(t'))/max(Fri'*c/sum(Fri)-Fro'*cv/sum(Fro)))]*0.55*6000/(nnz(Roi)*Grid.volume);
max(Fri'*c/sum(Fri))
[Pfit1,con1, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 2);
Fr(4) = Pfit1.BF;
[Pfit2,con2, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 1);
Fr(5) = Pfit2.BF;
[Pfit3,con3, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 3);
Fr(6) = Pfit3.BF;
[Pfit4,con4, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 4);
Fr(7) = Pfit4.BF;
Fd = (Fr-Fr(1))/Fr(1)*100
BF {jk} = [Fr;Fd];

%% Plot figure
folder = ['Result/EvalROI/',date,'/'];
[~,~,~] = mkdir(folder);
pathsave = [folder,'/Reg',datestr(now,'ddmmyy_HHMMSS'),'.png'];
msg = ['Saving ' pathsave]; disp(msg);

pathsave = [folder,'/4Model',datestr(now,'ddmmyy_HHMMSS'),'.png'];
HM = figure('name',['Model Fitting'],'Position',[100 100 1000 800]);
if any(gambar==jk)
figure(HM)
clf;
subplot(2,3,1)
imagesc(maXc,[0 max(maXc(:))/10000]);colormap(bone);axis image;axis off; title(['ROI']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
% plot(GV,'c','YData',vnode.XData/prm.pixel,'XData',vnode.YData/prm.pixel,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% plot(GA,'r','YData',node.XData/prm.pixel,'XData',node.YData/prm.pixel,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1]);

subplot(2,3,[4,5,6])
plot(t,roi,'k-',t,con3,'c:','LineWidth',2); title(['Signal fitting comparison']);
hold on
plot(t,con2,'g-.',t,con1,'m:','LineWidth',2);
plot(t,con4,'--','Color',[0,0.7,0.9],'LineWidth',2);
xlim([0 60])
xlabel('Time[s]')
ylabel('Concentration[mM]')
legend('ROI signal','MS fitting','2CM Uptake fitting','2CM Exchange fitting','1CM fitting');
set(gcf, 'Color', 'w');

subplot(2,3,[2,3])
sz = 20;
plot([min(Fr),max(Fr)],[min(Fr),max(Fr)],'k','LineWidth',1);
hold on
plot(Fr(1),Fr(2),'r*','LineWidth',2);
plot(Fr(1),Fr(3),'b*','LineWidth',2);
plot(Fr(1),Fr(4),'m*','LineWidth',2);
plot(Fr(1),Fr(5),'g*','LineWidth',2);
plot(Fr(1),Fr(6),'c*','LineWidth',2);
plot(Fr(1),Fr(7),'*','Color',[0,0.7,0.9],'LineWidth',2);
xlabel('Flow model [ml/100ml/min]')
ylabel(['Prediction plasma flow [ml/100ml/min]'])
legend('Identity line','MS1','MS analytic','2CM Uptake fitting','2CM Exchange fitting','MS fitting','1CM fitting');
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
end
%% AIF roots
b = segment.num(segment.isroot);
Fri = x(Co+b);
c=[];
for i = 1:length(b)
    c(i,:) = mean(cISave(Co+NCs+segment.Cnum{b(i)}(:),1:lim),1); 
end
Fg = [F,max(diff(N)./diff(t'))/max(Fri'*c/sum(Fri))]*0.55*6000/(nnz(Roi)*Grid.volume);

[Pfit1,con1, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 2);
Fg(3) = Pfit1.BF;
[Pfit2,con2, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 1);
Fg(4) = Pfit2.BF;
[Pfit3,con3, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 3);
Fg(5) = Pfit3.BF;
[Pfit4,con4, resnorm] = funfit(t, (Fri'*c/sum(Fri))', roi',Roi,Grid, 4);
Fg(6) = Pfit4.BF;
Fd = (Fg-Fg(1))/Fg(1)*100
BG {jk} = [Fg;Fd];

if any(gambar==jk)
pathsave = [folder,'/AIF2Model',datestr(now,'ddmmyy_HHMMSS'),'.png'];
HM = figure('name',['Model Fitting'],'Position',[100 100 1000 800]);
figure(HM)
clf;
subplot(2,3,1)
imagesc(maXc,[0 max(maXc(:))/10000]);colormap(bone);axis image;axis off; title(['ROI']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
% plot(GV,'c','YData',vnode.XData/prm.pixel,'XData',vnode.YData/prm.pixel,'LineWidth',GV.Edges.width,'DisplayName','Venal tree structure');
% plot(GA,'r','YData',node.XData/prm.pixel,'XData',node.YData/prm.pixel,'LineWidth',GA.Edges.width,'DisplayName','Arterial tree structure');
daspect([1 1 1]);

subplot(2,3,[4,5,6])
plot(t,roi,'k-',t,con3,'g:','LineWidth',2); title(['Signal fitting comparison']);
hold on
plot(t,con2,'m-.',t,con1,'b:','LineWidth',2);
plot(t,con4,'c--','LineWidth',2);
xlim([0 60])
xlabel('Time[s]')
ylabel('Concentration[mM]')
legend('ROI signal','MS fitting','2CM Uptake fitting','2CM Exchange fitting','1CM fitting');
set(gcf, 'Color', 'w');

subplot(2,3,[2,3])
sz = 20;
plot([min(Fg),max(Fg)],[min(Fg),max(Fg)],'k','LineWidth',1);
hold on
plot(Fg(1),Fg(2),'r*','LineWidth',2);
plot(Fg(1),Fg(3),'b*','LineWidth',2);
plot(Fg(1),Fg(4),'m*','LineWidth',2);
plot(Fg(1),Fg(5),'g*','LineWidth',2);
plot(Fg(1),Fg(6),'c*','LineWidth',2);
xlabel('Flow model [ml/100ml/min]')
ylabel(['Prediction plasma flow [ml/100ml/min]'])
legend('Identity line','MS1','2CM Uptake fitting','2CM Exchange fitting','MS fitting','1CM fitting');
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
end
bfn(jk,:,nm)=round(BF{jk}(1,:)',2);
bfe(jk,:,nm)=round(BF{jk}(2,:)',2);
bgn(jk,:,nm)=round(BG{jk}(1,:)',2);
bge(jk,:,nm)=round(BG{jk}(2,:)',2);
end
% Recapitulate data
[bf,bg] = Rekap(BF,BG)
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
% average error
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
% estimation
for j=1:6
bg=[];
for i = 1:11
    bg(:,i)=round(BG{i}(1,:)',2);
end
end
bg(:,end+1) = ab;
% error
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
% estimation
for j=1:7
bf=[];
for i = 1:11
    bf(:,i)=round(BF{i}(1,:)',2);
end
end
bf(:,end+1) = ab;
end
% end