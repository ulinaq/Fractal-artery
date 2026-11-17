function val = FOptim(var)
% function of x (perturbation on tree structure) to compute a value for
% optimization solver
% global node segment vnode vsegment Grid Trans prm;
prm = Frog_parameter();
load('initial.mat');
%Update structure by x perturbation
[nod,sgm]=FO_update(var(1),node,segment);
[vnod,vsgm]=FO_update(var(2),vnode,vsegment);
%Solve darcy domain
[state,Grid,nod,vnod] = FO_Capillary(nod,vnod,Grid);
%Solve system of equation
x = Frog_LinEq(prm,nod,sgm,vnod,vsgm,state.P);
Co = nod.n + vnod.n;
Cv = Co + sgm.n;
seg = 1:1:Co;
in = x([Co+seg(sgm.isterminal)]);
out = x([Cv+seg(vsgm.isterminal)]);
q = [in/sum(in)*sum(out);-1*out];
Pmat = x(Co)+state.pvec*q;
Pmat = reshape(Pmat,Grid.Nx,Grid.Ny);

aa = 1-Grid.K;
aa = reshape(aa,Grid.N,1);

P = ones(Grid.Nx,Grid.Ny)*min(min(Pmat));
B = logical(Grid.K);
P(B) = Pmat(B);                                         
%Compute time of flight
V = Flux(P,Grid,Trans,q,aa);
Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);

val=norm((1./V.u(B)-mean(1/V.u(B))),2);
end
