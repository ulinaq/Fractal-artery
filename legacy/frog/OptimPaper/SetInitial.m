function SetInitial(n, dLength)
% %% Domain Parameters (preparation)
name = 'TOF-result';
prm = Frog_parameter();
[Trans,Grid] = FO_Darcy(prm);    % Load the image domain and create darcy domain. Then compute transmibility
[node,segment,vnode,vsegment] = SetVessel2(prm, Grid, dLength, n);
i = 0;
while all(Grid.B([node.input; vnode.output])) == 0
    [node,segment,vnode,vsegment] = SetVessel2(prm, Grid, dLength, n);
    i = i + 1
end
[Xr,Grid] = FO_Capillary(node,vnode,Grid);
x = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% x = Xr.pvec(2*Grid.N+1:end);
Co = node.n + vnode.n;
Cv = Co + segment.n;
in = x([Co+segment.num(segment.isterminal)]);
out = x([Cv+vsegment.num(vsegment.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
P = Xr.pvec*q+x(end);
Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);

% add flow rate as weight on graph plot
GV=graph(vsegment.init,vsegment.end,abs(x(Cv+1:Cv+vsegment.n)));
GA=graph(segment.init,segment.end,abs(x(Co+1:Cv)));
GV.Edges.width=10*sqrt(abs(GV.Edges.Weight)/sum(out));
GA.Edges.width=10*sqrt(abs(GA.Edges.Weight)/sum(in));
% Compute Time of Flight
% V = FluxSingle(prm,Pa,Grid,Trans,q,node,vnode,segment,x);
V = Flux0(Pa,Grid,Trans,q);
mean(V.Ti(Grid.B))
% save init.mat node segment vnode vsegment Grid Trans;
%% Tree visualization
analyze(prm,Pa,V,Grid,GA,GV,node,vnode, name)

function [node,segment,vnode,vsegment] = SetVessel2(prm, Grid, dLength, n)
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
var =  dLength*(rand(1,n)-0.5);
[node,segment]=FO_update(var(1:round(n/4)),var(round(n/4)+1:round(n/2)),node,segment,Grid);
[vnode,vsegment]=FO_update(var(round(n/2)+1:round(3*n/4)), var(round(3*n/4)+1:end),vnode,vsegment,Grid);
end 

function [node,segment,vnode,vsegment] = SetVessel(prm, Grid, dLength, dTheta)
[node,segment,vnode,vsegment] = segOne(prm,Grid,0,0); 
% the initial vascular structure
dxA = dLength*(rand(1,node.n)-0.5);
dThetaA = dTheta*(rand(1,segment.n)-0.5);
dxV = dLength*(rand(1,vnode.n)-0.5);
dThetaV = dTheta*(rand(1,vsegment.n)-0.5);
%Update structure by x perturbation
[node,segment]=FO_update(dxA,dThetaA,node,segment,Grid);
[vnode,vsegment]=FO_update(dxV,dThetaV,vnode,vsegment,Grid);
end
end