figure, imshow(B); hold on
scatter(x2,y2); daspect([1 1 1])
a = [1:length(x2)]'; b = num2str(a); c = cellstr(b);
text(x2, y2, c);
par=[1,2;2,3;2,4;3,7;3,8;4,5;4,6;6,9;6,10;9,17;9,18;5,11;5,12;11,15;11,16;16,19;16,20;15,21;15,22;22,33;12,13;12,14;14,23;14,24;23,31;23,32;32,34;32,100;24,29;24,30;13,25;13,26;26,27;26,28;28,37;28,101;101,35;101,36;36,98;36,99;27,39;27,38;39,40;39,41;41,97;41,53;97,42;97,43;43,54;54,44;43,45;45,49;45,46;46,47;46,48;53,55;53,52;52,51;52,50;55,59;55,56;56,57;56,58;55,60;60,61;60,63;63,71;63,64;64,65;64,66;66,70;66,67;70,68;70,69;71,72;71,73;73,74;73,75;75,76;75,78;78,77;78,79;61,62;61,80;80,81;80,87;81,82;81,96;96,83;96,84;84,85;84,86;87,88;87,89;89,90;89,91;91,92;91,93;93,94;93,95];
pas=[1,2;2,3;2,4;3,7;3,8;4,5;4,6;5,14;5,13;6,12;6,11;8,10;8,9;11,22;11,21;12,20;12,19;13,18;13,17;14,15;14,16;15,28;15,27;16,25;16,26;21,24;21,23;26,34;26,33;27,32;27,31;28,29;28,30;30,42;30,41;31,40;31,39;32,38;32,37;34,36;34,35;41,46;41,45;42,44;42,43;43,52;43,51;44,50;44,49;45,48;45,47;47,58;47,57;51,56;51,55;52,54;52,53;55,60;55,59;56,62;56,61];
seg.init=[pas(:,1)]
seg.end=[pas(:,2)];
parent=zeros(62,1);
parent(pas(:,2))=pas(:,1);
nodes.Parent=parent;
junc=zeros(62,1);
junc(pas(:,1))=1;
nodes.junction=junc;
nodes.n=length(x);
nodes.XData=x;
nodes.YData=y;

vseg.init=[par(:,1)]
vseg.end=[par(:,2)];
parent=zeros(101,1);
parent(par(:,2))=par(:,1);
vnodes.Parent=parent;
junc=zeros(101,1);
junc(par(:,1))=1;
vnodes.junction=junc;
vnodes.n=length(x2);
vnodes.XData=x2;
vnodes.YData=y2;

GA=graph(seg.init,seg.end);
GV=graph(vseg.init,vseg.end);
figure, imshow(B); hold on
plot(GA,'k','XData',nodes.XData,'YData',nodes.YData);
title('Vein Tree');


figure, imshow(B); hold on
plot(GV,'c','XData',vnodes.XData,'YData',vnodes.YData);
title('Vein Tree');
for i=1:length(pas)
    x1=x(seg.init(i));
    y1=y(seg.init(i));
    x2=x(seg.end(i));
    y2=y(seg.end(i));
    seg.len(i) = norm([x2-x1,y2-y1]);
P0 = [x1, y1]; 
P1 = [x1, y1+1];
P2 = [x2, y2];
n1 = (P2 - P0) / norm(P2 - P0);  % Normalized vectors
n2 = (P1 - P0) / norm(P1 - P0);
seg.theta(i) = atan2(norm(det([n2; n1])), dot(n1, n2));

end
nodes.no=1:62;
rad=zeros(61,1);
rad(nodes.no(junc==0)-1)=1;
rad([28,52,58,49,55,59,47,46,45,30,55,34:30,16,11,20,7,10])=1.3;
rad([5,2,12,15,25,26,40,44,43,33,31,])=2;
rad([54,51])=2.2; rad([50,42,41,29,27,14,13,4,3,1])=[2.9,3.1,3.3,3.5,3.7,3.8,4,4.2,4.5,5];
seg.rad=rad;
seg.n=61;

for i=1:length(par)
    x1=vnodes.XData(vseg.init(i));
    y1=vnodes.YData(vseg.init(i));
    x2=vnodes.XData(vseg.end(i));
    y2=vnodes.YData(vseg.end(i));
    vseg.len(i) = norm([x2-x1,y2-y1]);

P0 = [x1, y1]; 
P1 = [x1, y1+1];
P2 = [x2, y2];
n1 = (P2 - P0) / norm(P2 - P0);  % Normalized vectors
n2 = (P1 - P0) / norm(P1 - P0);
vseg.theta(i) = atan2(norm(det([n2; n1])), dot(n1, n2));
end
vnodes.no=1:101;
rad=zeros(100,1);
rad(vnodes.no(junc==0)-1)=1;
rad([2,8,15,21,32,23,35,98,46,48,66,51,55,67,69,53,77,41,71,31,22,83,81,87,78])=1.5;
rad([95,81,72,65,44,74,100,27,5,13,14])=1.8; rad([45,27,10,42,63,70,80,94])=2; rad([92,62,96,39])=2.4;
rad([90,88,86,79,60,59,54,52,40,38,26,25,12,11,4,3,1])=[2.7,2.8,2.9,3.1,3.3,3.7,3.9,4.1,4.3,4.4,4.7,5,5.4,5.8,6.2,6.6,7];
vseg.rad=rad;
vseg.n=100;

button(button==48)=1;
button(button==49)=2;
button(button==50)=3;
button(button==51)=4;
button(button==52)=5;
button(button==53)=6;
button(button==54)=7;
button(button==55)=8;
button(button==56)=9;
button(button==57)=10;
button(button==113)=11;
button(button==119)=12;
button(button==101)=13;

node=nodes; vnode=vnodes; segment=seg; vsegment=vseg;
%% Compute flow
%% Capillary Domain
% domain size 
[state,T,Grid,node,vnode] = Frog_Capillary(node, vnode);

%% Calculate pressure in nodes (Update April 26, 2018)
prm.Rho = 1;
prm.mu = 1;
prm.Pin = 1e6;           % capillary pressure
prm.Qin = 100;          % Flowrate in input node
xli = Tree_LinEq(prm,nodes,seg,vnodes,vseg,state.P);
Co = node.n + vnode.n;
Cv = Co + segment.n;
seg = 1:1:Co;
q = [xlin([Co+seg(node.in)-1]);-1*xlin([Cv+seg(vnode.out)-1])];
Pmat = xlin(Co)+state.pvec*q;
Pmat = reshape(Pmat,Grid.Ny,Grid.Nx);
state.pressure = Pmat;
V = Flux(Pmat,Grid,T,q);

Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);

figure()
contourf(Ti');  colorbar; title('Time of Flight');
daspect([1 1 1])

figure()
contourf(Pmat,20); colorbar; title('Pressure');
hold on;
quiver(V.qx',V.qy','r'); %title('Fluxes');
toc

% clear Co Cv q seg Pmat

%% Calculate Flux
function V = Flux(Pmat,Grid,Trans,q)
   % Fluxes at staggered grid
   Ny=Grid.Ny; Nx=Grid.Nx; N=Ny*Nx;
       
   V.y = zeros(Nx,Ny+1);
   V.x = zeros(Nx+1,Ny);;
   V.y(:,2:Ny) = (Pmat(:,1:Ny-1)-Pmat(:,2:Ny)).*Trans.y(:,2:Ny);
   V.x(2:Nx,:) = (Pmat(1:Nx-1,:)-Pmat(2:Nx,:)).*Trans.x(2:Nx,:);

   V.qy = movmean(V.y,2,2,'Endpoints','discard');
   V.qx = movmean(V.x,2,1,'Endpoints','discard');
   V.u = sqrt(V.qx.^2+V.qy.^2);
   u = reshape(V.u,N,1);
   V.range = max(u)-min(u);
   
   qq=[];
for i=1:length(q)
    qq=[qq; Grid.src.rate{i}*q(i)];
end

qi = sparse(Grid.src.cell,1, qq, N,1);
pv = reshape(Grid.volume.*Grid.poro,N,1);
% define neighbor
num = reshape([1:N],Nx,Ny);
Yin = [zeros(Nx,1) num];
Yout = [num zeros(Nx,1)];
Xin = [zeros(1,Ny); num];
Xout = [num; zeros(1,Ny)];
face.neighbors = [reshape(Xin,[],1), reshape(Xout,[],1);reshape(Yin,[],1), reshape(Yout,[],1)];
face.flux = [reshape(V.x,[],1);reshape(V.y,[],1)];
ind = ~any(face.neighbors==0,2);
n  = double(face.neighbors(ind,:));

out = min(face.flux(ind), 0);
in  = max(face.flux(ind), 0);

% Cell wise total inflow
inflow  = accumarray([n(:, 2); n(:, 1)], [in; -out]);

% diagonal 
d = inflow + max(qi,0);

% Inflow flux matrix
A  = sparse(n(:,2), n(:,1),  in, N, N)...
   + sparse(n(:,1), n(:,2), -out, N, N);
A = -A + spdiags(d, 0, N, N);

% Solve
V.Ti = A\pv;
  
end

