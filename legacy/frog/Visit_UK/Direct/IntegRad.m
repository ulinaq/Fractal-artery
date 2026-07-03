% function [N] = integralC(Grid,node,segment,vnode,vsegment,Connect,coSave,x,Time,ROI)
Co = node.n + vnode.n;  Cv = Co + segment.n;    Cq = Cv+vsegment.n; 
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz); % create meshgrid

% Concentration array time series
lim = size(coSave,2)-1;
coS = coSave(:,1:lim);
maXc = full(reshape(max(coS,[],2),Grid.Nx,Grid.Ny));
Vol = Comp(Grid,node,vnode,segment,vsegment,Connect);

tic;
Reps = 100;
Mn = 50;
Bflow = zeros(8,Mn);
BflowG = zeros(8,Mn);
for jk = 1:Mn
Rx = randi([Reps Grid.Nx-Reps]);
Ry = randi([Reps Grid.Ny-Reps]);
Center = sub2ind([Grid.Nx Grid.Ny],Rx,Ry);
Roi = (xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Reps^2;
% Roi = logical(Froi{jk});

% ROI choosen 
Rper = bwmorph(Roi,'thin',1);
% Skeleton mapping segment
Skel = SkelAll(Grid,node,vnode,segment,vsegment,Connect);
roi = full(mean(coS(Roi,1:lim)));
% Find vascular segment and ROI intersection 
[indA,sIn,indV,sOut,indC] = Sections(node,vnode,segment,vsegment,Connect,Skel,Rper,x);
Fri = x([Co+sIn.A;Cv+sIn.V;Cq+sIn.C]);

%% Evaluate real flow
F = sum(Fri)*0.55*6000/(nnz(Roi)*Grid.volume);
BVreal = mean(Vol(Roi))*100*0.55;
%% localized AIF 
c=[];   
for i = 1:length(sIn.A)
    c(i,:) = mean(cISave(Co+NCs+segment.Cnum{sIn.A(i)}(sIn.indA{i}),1:lim),1); 
end
if ~isempty(sIn.V)
    j=1;
for i = length(sIn.A)+1:length(sIn.A)+length(sIn.V)
    c(i,:) = mean(cISave(Co+NCs+NAs+vsegment.Cnum{sIn.V(j)}(sIn.indV{j}),1:lim),1); 
    j = j+1;
end
end 
if ~isempty(sIn.C)
    j = 1;
for i = length(sIn.A)+length(sIn.V)+1:length(sIn.A)+length(sIn.V)+length(sIn.C)
    c(i,:) = mean(cISave(node.n+Connect.Cnum{sIn.C(j)}(sIn.indC{j}),1:lim),1); 
    j = j+1;
end
end 
aif = (Fri'*c/sum(Fri))';
%% AIF roots
% aif = full(def_AIF(coS(:,1:lim),segment,Grid,4))';
Fg = ones(5,1)*F;
if F~=0
    F=round(F); BVreal=round(BVreal);
%% Fitting data using pharmakokinetic models
[Pfit1,con1, resnorm] = funfit(t, aif, roi', 2);% 2CM X model
Fg(2) = Pfit1.BF;
[Pfit2,con2, resnorm] = funfit(t, aif, roi', 1);% 2CM uptake model
Fg(3) = Pfit2.BF;
[Pfit3,con3, resnorm] = funfit(t, aif, roi', 3,F,BVreal);% Maximum Slope model
Fg(4) = Pfit3.BF;
[Pfit4,con4, resnorm] = funfit(t, aif, roi', 4,F,BVreal);% Compartment model
Fg(5) = Pfit4.BF;
Fv = [BVreal,Pfit1.BV,Pfit2.BV];
Bflow(:,jk) = [Fg;Fv'];
end
% if abs(Bflow(1,jk)-Bflow(2,jk))<0.2*Bflow(1,jk)
%% Plot figure
folder = ['Result/EvalROIrad/',date,'/'];
[~,~,~] = mkdir(folder);
% if abs(Bflow(1,jk)-Bflow(2,jk))<0.2*Bflow(1,jk)
pathsave = [folder,'/Radius',num2str(Reps),datestr(now,'ddmm_HHMMSS'),'.png'];
HM = figure('name',['Model Fitting'],'Position',[100 100 1000 700]);
figure(HM)
clf;
subplot(2,3,1)
imagesc(maXc,[0 max(maXc(:))/10000]);colormap(bone);axis image;axis off; title(['ROI']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
daspect([1 1 1]);

subplot(2,3,[2,3])
plot(t,roi,'k-',t,con3,'g:','LineWidth',2); title(['Signal fitting comparison']);
hold on
plot(t,con4,'c-.',t,con2,'b:','LineWidth',2);
plot(t,con1,'m--','LineWidth',2);
xlim([0 60])
xlabel('Time[s]')
ylabel('Concentration[mM]')
legend('ROI signal','MS','1CM','2CM Uptake','2CM Exchange');
set(gcf, 'Color', 'w');

subplot(2,12,[14,15,16,17,18])
plot([min(Fg),max(Fg)],[min(Fg),max(Fg)],'k','LineWidth',1);title(['Blood flow comparison']);
hold on
plot(Fg(1),Fg(4),'g*','LineWidth',2);
plot(Fg(1),Fg(5),'c*','LineWidth',2);
plot(Fg(1),Fg(3),'b*','LineWidth',2);
plot(Fg(1),Fg(2),'m*','LineWidth',2);
xlabel('Flow model [ml/100ml/min]')
ylabel(['Plasma flow prediction [ml/100ml/min]'])
legend({'Identity line','MS','1CM','2CM Uptake','2CM Exchange'},'Location','northwest');
set(gcf, 'Color', 'w');

subplot(2,12,[20,21,22,23,24])
plot([min(Fv),max(Fv)],[min(Fv),max(Fv)],'k','LineWidth',1);title(['Blood volume comparison']);
hold on
plot(Fv(1),Fv(3),'b*','LineWidth',2);
plot(Fv(1),Fv(2),'m*','LineWidth',2);
xlabel('Blood volume model [ml/100ml]')
ylabel(['Blood volume prediction [ml/100ml]'])
legend({'Identity line','2CM Uptake','2CM Exchange'},'Location','northwest');
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
% end

%% AIF roots
aif = full(def_AIF(coS(:,1:lim),segment,Grid,4))';
Fg = ones(5,1)*F;
if F~=0
    F=round(F); BVreal=round(BVreal);
%% Fitting data using pharmakokinetic models
[Pfit1,con1, resnorm] = funfit(t, aif, roi', 2);% 2CM X model
Fg(2) = Pfit1.BF;
[Pfit2,con2, resnorm] = funfit(t, aif, roi', 1);% 2CM uptake model
Fg(3) = Pfit2.BF;
[Pfit3,con3, resnorm] = funfit(t, aif, roi', 3,F,BVreal);% Maximum Slope model
Fg(4) = Pfit3.BF;
[Pfit4,con4, resnorm] = funfit(t, aif, roi', 4,F,BVreal);% Compartment model
Fg(5) = Pfit4.BF;
Fv = [BVreal,Pfit1.BV,Pfit2.BV];
BflowG(:,jk) = [Fg;Fv'];
end
% if abs(Bflow(1,jk)-Bflow(2,jk))<0.2*Bflow(1,jk)
%% Plot figure
folder = ['Result/EvalROIrad/',date,'/'];
[~,~,~] = mkdir(folder);
% if abs(Bflow(1,jk)-Bflow(2,jk))<0.2*Bflow(1,jk)
pathsave = [folder,'/Radius',num2str(Reps),datestr(now,'ddmm_HHMMSS'),'.png'];
HM = figure('name',['Model Fitting'],'Position',[100 100 1000 700]);
figure(HM)
clf;
subplot(2,3,1)
imagesc(maXc,[0 max(maXc(:))/10000]);colormap(bone);axis image;axis off; title(['ROI']);daspect([1 1 1])
hold on
imagesc(Roi);daspect([1 1 1]);alpha(0.6-0.2*Roi)
daspect([1 1 1]);

subplot(2,3,[2,3])
plot(t,roi,'k-',t,con3,'g:','LineWidth',2); title(['Signal fitting comparison']);
hold on
plot(t,con4,'c-.',t,con2,'b:','LineWidth',2);
plot(t,con1,'m--','LineWidth',2);
xlim([0 60])
xlabel('Time[s]')
ylabel('Concentration[mM]')
legend('ROI signal','MS','1CM','2CM Uptake','2CM Exchange');
set(gcf, 'Color', 'w');

subplot(2,12,[14,15,16,17,18])
plot([min(Fg),max(Fg)],[min(Fg),max(Fg)],'k','LineWidth',1);title(['Blood flow comparison']);
hold on
plot(Fg(1),Fg(4),'g*','LineWidth',2);
plot(Fg(1),Fg(5),'c*','LineWidth',2);
plot(Fg(1),Fg(3),'b*','LineWidth',2);
plot(Fg(1),Fg(2),'m*','LineWidth',2);
xlabel('Flow model [ml/100ml/min]')
ylabel(['Plasma flow prediction [ml/100ml/min]'])
legend({'Identity line','MS','1CM','2CM Uptake','2CM Exchange'},'Location','northwest');
set(gcf, 'Color', 'w');

subplot(2,12,[20,21,22,23,24])
plot([min(Fv),max(Fv)],[min(Fv),max(Fv)],'k','LineWidth',1);title(['Blood volume comparison']);
hold on
plot(Fv(1),Fv(3),'b*','LineWidth',2);
plot(Fv(1),Fv(2),'m*','LineWidth',2);
xlabel('Blood volume model [ml/100ml]')
ylabel(['Blood volume prediction [ml/100ml]'])
legend({'Identity line','2CM Uptake','2CM Exchange'},'Location','northwest');
set(gcf, 'Color', 'w');
export_fig(pathsave,'-r300');
% end
end
% end
% [~,tem] = find(Bflow==0);
% Bflow(:,unique(tem))=[];
b =zeros(size(Bflow));
for i=1:length(Bflow)
b(1:5,i)=Bflow(1:5,i)/Bflow(1,i);
b(6:8,i)=Bflow(6:8,i)/Bflow(6,i);
end
[~,tem] = find(b>100 | b<0.01);
b(:,unique(tem))=[];
Flmean = mean(b,2)
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

% Compute volume fraction/Porosity
function Vol = Comp(Grid,node,vnode,segment,vsegment,Connect)
cA = sparse(Grid.Nx,Grid.Ny); cV = sparse(Grid.Nx,Grid.Ny); cO = sparse(Grid.Nx,Grid.Ny);
Vol = zeros(Grid.Nx,Grid.Ny);
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

function area = CircleROI(xcell,ycell,Rep,Grid)
        Reps = 1*Rep;
        Rx = randi([Reps Grid.Nx-Reps]);
        Ry = randi([Reps Grid.Ny-Reps]);
        Center = sub2ind([Grid.Nx Grid.Ny],Rx,Ry);
        area=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Rep^2;
    end
