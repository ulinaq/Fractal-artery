% Numerical experiments for Non-linearity comparison and 50% occlusions
% 

tStart = tic;
prm = Frog_parameter();                                 % Load the modelling parameters
[Trans,Grid] = FO_Darcy3(prm);                          % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);   % Load the segmented data and modification
[Xr,Grid] = FO_Capillary2(node,vnode,Grid);             % Solve Darcy and coupling
Xr.Pext = Xr.C([segment.ext; vsegment.ext],:);          % Arrange the coupling to the vascular network
xlin = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);    % Solve VG linear system
Co = node.n + vnode.n;  Cv = Co + segment.n;
tPre = toc(tStart);
% Compute the baseline model pressure distribution for comparison
xbase = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,xlin);
in = xbase([Co+segment.num(segment.isterminal)]);
out = xbase([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
Pbase = Xr.C*q+xbase(end);
for i = 1:12
    name = ['_type_',num2str(i)];
    switch i
        case 1  % baseline model
            name = ['Baseline'];
            tic
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,xlin);
            Tbl.T(i) = toc;
        case 2  % Elastic model
            name = ['Elastic'];
            tic
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,2,xlin);
            Tbl.T(i) = toc;
        case 3  % Junction model
            name = ['Junction'];
            tic
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,3,xlin);
            Tbl.T(i) = toc;
        case 4  % linear
            name = ['Linear'];
            tic
            x = xlin;
            Tbl.T(i) = toc;
        case 5  % 'Artery-1 occlusion'
            name = ['Artery-1-Occlusion'];
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
%             segment.rad(120)=segment.rad(120)/2;
            segment.rad(74)=segment.rad(74)/2;   % segment that is damaged has much smaller radii
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,[]);
            Tbl.T(i) = toc;
            in = x([Co+segment.num(segment.isterminal)]); out = x([Cv+vsegment.num(vsegment.isterminal)]);
            q = [in/sum(in)*sum(out);-1*out];
            P = Xr.C*q+x(end)-Pbase;
            GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
            GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
            GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
            GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
            analyzex(prm,P,Grid,GA,GV,node,vnode,name);            
        case 6  % 'Artery-2 occlusion'
            name = ['Artery-2-Occlusion'];
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            segment.rad(85)=segment.rad(85)/2;   % segment that is damaged has much smaller radii
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,[]);
            Tbl.T(i) = toc;
            in = x([Co+segment.num(segment.isterminal)]); out = x([Cv+vsegment.num(vsegment.isterminal)]);
            q = [in/sum(in)*sum(out);-1*out];
            P = Xr.C*q+x(end)-Pbase;
            GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
            GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
            GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
            GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
            analyzex(prm,P,Grid,GA,GV,node,vnode,name);
        case 7  % 'Vein-1 occlusion'
            name = ['Vein-1-Occlusion'];
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            vsegment.rad(223)=vsegment.rad(223)/2;
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,[]);
            Tbl.T(i) = toc;
            in = x([Co+segment.num(segment.isterminal)]); out = x([Cv+vsegment.num(vsegment.isterminal)]);
            q = [in/sum(in)*sum(out);-1*out];
            P = Xr.C*q+x(end)-Pbase;
            GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
            GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
            GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
            GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
            analyzex(prm,P,Grid,GA,GV,node,vnode,name);
        case 8  % 'Vein-2 occlusion'
            name = ['Vein-2-Occlusion'];
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            vsegment.rad(173)=vsegment.rad(173)/2;   % segment that is damaged has much smaller radii
            x = Frog_nonlinEq3(node,prm,segment,vnode,vsegment,Xr,1,[]);
            Tbl.T(i) = toc;
            in = x([Co+segment.num(segment.isterminal)]); out = x([Cv+vsegment.num(vsegment.isterminal)]);
            q = [in/sum(in)*sum(out);-1*out];
            P = Xr.C*q+x(end)-Pbase;
            GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
            GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
            GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
            GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
            analyzex(prm,P,Grid,GA,GV,node,vnode,name);
%% Linear model occlusions
        case 9  % 'Artery-1 occlusion'
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            segment.rad(74)=segment.rad(74)/2;   % segment that is damaged has much smaller radii
            x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
            Tbl.T(i) = toc;
        case 10  % 'Artery-2 occlusion'
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            segment.rad(85)=segment.rad(85)/2;   % segment that is damaged has much smaller radii
            x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
            Tbl.T(i) = toc;
        case 11  % 'Vein-1 occlusion'
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            vsegment.rad(223)=vsegment.rad(223)/2;
            x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
            Tbl.T(i) = toc;
        case 12  % 'Vein-2 occlusion'
            tic
            [node,segment,vnode,vsegment] = segOne(prm,Grid,0,0);
            vsegment.rad(173)=vsegment.rad(173)/2;   % segment that is damaged has much smaller radii
            x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
            Tbl.T(i) = toc;
    end
    tic
    in = x([Co+segment.num(segment.isterminal)]);
    out = x([Cv+vsegment.num(vsegment.isterminal)]);
    q = [in/sum(in)*sum(out);-1*out];
    P = Xr.C*q+x(end);
    Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
    Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);
    Var{i} = Itungan(x,prm,Grid,node,segment,vnode,Pa,Pv);
    GV=graph(vsegment.init,vsegment.end,abs(x(Cv+vsegment.num)));
    GA=graph(segment.init,segment.end,abs(x(Co+segment.num)));
    GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
    GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
    V = Flux(prm,Pa,Pv,Grid,Trans,q,node,vnode,segment,x);  % Fluxes in Darcy
    analyze(prm,P,V,Grid,GA,GV,node,vnode,name);            % Pressure distribution in Darcy domain
    Tbl.Ttotal(i) = Tbl.T(i) + tPre + toc;
end

%% Create the recapitulation table
for i = 1:12
   Tbl.Q(i) = Var{i}.Q; 
   Tbl.RTot(i) = Var{i}.RTot; 
   Tbl.RAtree(i) = Var{i}.RAtree; 
   Tbl.RVtree(i) = Var{i}.RVtree; 
   Tbl.RAcap(i) = Var{i}.RAcap; 
   Tbl.RVcap(i) = Var{i}.RVcap ; 
end
% 
Tbl.Sim = {'Baseline model';'Elasticity model';'Junction model';'Linear model';...
    'Artery-1 occlusion';'Artery-2 occlusion';'Vein-1 occlusion';'Vein-2 occlusion';...
    'Artery-1 occlusion linear';'Artery-2 occlusion linear';'Vein-1 occlusion linear';'Vein-2 occlusion linear'};
Tbl.VarNames = {'Total Blood Flow [mm^3/s]',...
    'Total Resistance[kg/mm^4.s]','R_{aN}','R_vN','R_{aC}','R_{vC}','t_{tot} [s]'};
T = table(Tbl.Q',Tbl.RTot',Tbl.RAtree',Tbl.RVtree',Tbl.RAcap',Tbl.RVcap',Tbl.Ttotal','RowNames',Tbl.Sim);
H = figure('name',['Numerical simulation results in several experiments'],'Position',[300 300 1050 300]);
figure(H);
uitable('Data',T{:,:},'ColumnName',Tbl.VarNames,'RowName',T.Properties.RowNames,'Position',[40 20 957 266]);
set(gcf, 'Color', 'w');
asd = [Tbl.Q',Tbl.RTot',Tbl.RAtree',Tbl.RVtree',Tbl.RAcap',Tbl.RVcap',Tbl.Ttotal']

tEnd = toc(tStart);

function [Var] = Itungan(x,prm,Grid,node,segment,vnode,Pa,Pv)
    Var.x=x;
    Co = node.n + vnode.n; Cv = Co+segment.n;
    Var.Q = sum(x([Co+segment.num(segment.isroot)]));
    Var.RTot = (prm.Pin-prm.Pout)/Var.Q;
    dPa = mean(x(node.isterminal));
    dPv = mean(x(node.n+vnode.num(vnode.isterminal)));
    dPC = 0.5*(mean(Pv(Grid.B))+ mean(Pa(Grid.B)));
    Var.RAtree = (prm.Pin-dPa)/(prm.Pin-prm.Pout)*Var.RTot;
    Var.RAcap = std(Pa(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*Var.RTot;
    Var.RVcap = std(Pv(Grid.B))/(std(Pa(Grid.B))+std(Pv(Grid.B)))*(dPa-dPv)/(prm.Pin-prm.Pout)*Var.RTot;
    Var.RVtree = (dPv-prm.Pout)/(prm.Pin-prm.Pout)*Var.RTot;
end


