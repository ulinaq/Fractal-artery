function val = FOptim(var)
% function of x (perturbation on tree structure) to compute a value for
% optimization solver
% global node segment vnode vsegment Grid Trans prm;
prm = Frog_parameter();             
% Load the image domain and create darcy domain. Then compute transmibility
load('init.mat');
% Update structure by x perturbation
n = length(var);
[nod,sgm]=FO_update(var(1:round(n/4)),var(round(n/4)+1:round(n/2)),node,segment,Grid);
[vnod,vsgm]=FO_update(var(round(n/2)+1:round(3*n/4)), var(round(3*n/4)+1:end),vnode,vsegment,Grid);
% [nod,sgm]=FO_update(var(1:node.n),var(node.n+1:node.n+segment.n),node,segment,Grid);
% [vnod,vsgm]=FO_update(var(node.n+segment.n+1:node.n+segment.n+vnode.n), var(node.n+segment.n+vnode.n+1:end),vnode,vsegment,Grid);
%Solve darcy domain
if all(Grid.B([node.input; vnode.output]))
    [Xr,Grid] = FO_Capillary2(nod,vnod,Grid); 
    Xr.Pext = Xr.C([sgm.ext; vsgm.ext],:);
    % Solve the linear system
    x = Frog_LinEq(prm,nod,sgm,vnod,vsegment,Xr.P);
    Co = node.n + vnode.n;  Cv = Co + segment.n;
    in = x([Co+sgm.num(sgm.isterminal)]);
    out = x([Cv+vsegment.num(vsgm.isterminal)]);
    q = [in/sum(in)*sum(out);-1*out];
    P = Xr.C*q+x(end);
    Pa = reshape(P(1:Grid.N),Grid.Nx,Grid.Ny);
    Pv = reshape(P(Grid.N+1:2*Grid.N),Grid.Nx,Grid.Ny);

    V = Flux(prm,Pa,Pv,Grid,Trans,q,nod,vnod,sgm,x); 
    
%     val=norm((1./V.u(Grid.B)-mean(1/V.u(Grid.B))),2);
%     val=std(V.Ti(Grid.B));
    val=std(V.u(Grid.B)); 
else
    val = inf;
end

end
