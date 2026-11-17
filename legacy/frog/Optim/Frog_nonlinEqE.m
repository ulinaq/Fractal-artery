function x = Frog_nonlinEqE(node,prm,segment,vnode,vsegment,Xr)
%% Junction or elasticity model 
% Non-linearities in the model (with Elasticity, Elas = 1; with Junction drop, Junc = 1)
% Type 1 is the nonlinear model with pressure drop at junctions
% Type 2 is the nonlinear model with radius dependence of pressure
% difference
% Type 3 is the combination of pressure drop at junction and elasticity
% model. Otherwise, it is the baseline model
% switch prm.type
Junc = 1; Elas = 1;

Pin = prm.Pin; Pout = prm.Pout;
Co = node.n + vnode.n; Cv = Co+segment.n;
eqCount = 1; j=1; k=1;
xend = Cv+vsegment.n+1;
in = node.num(node.isterminal); 
out = vnode.num(vnode.isterminal);
Nin = nnz(in); Nout = nnz(out);
eqCount = 1;

%% Equation on segment (pressure drop)
% Create Linear system for initial guess
guess = Frog_LinEq(prm,node,segment,vnode,vsegment,Xr.P);
% Artery 
for i = 1 : segment.n
    segint = segment.init(i);
    segend = segment.end(i);
    F{eqCount} = @(x) x(segint) - x(segend) - 8*prm.mu*segment.len(i)*x(Co+i)/(pi*segment.rad(i)^4)...
                *(1+(1-prm.Poison^2)*segment.rad(i)*(x(segint)/2 + x(segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
                + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.ha*prm.E))^(-4*Elas);
    eqCount = eqCount + 1; k=k+1;
end

%% Vein 
for i = 1 : vsegment.n
    segint = vsegment.init(i);
    segend = vsegment.end(i);
    F{eqCount} = @(x) x(node.n+segint) - x(node.n+segend) + 8*prm.mu*vsegment.len(i)*x(Cv+i)/(pi*vsegment.rad(i)^4)...
                *(1+(1-prm.Poison^2)*vsegment.rad(i)*(x(node.n+segint)/2 + x(node.n+segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
                + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.ha*prm.E))^(-4*Elas);
    eqCount = eqCount + 1; k=k+1;
end
%% Equation on node 
for i = 1 : node.n
    lsg = segment.num(segment.end==i);
    sg = segment.num(segment.init==i);
    if node.isroot(i)
        F{eqCount} = @(x) x(i) - Pin; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif node.isterminal(i)       % Equation on terminal nodes
        F{eqCount} = @(x) x(i) - x(xend) - prm.mu/prm.gamma*x(Co+lsg) - Xr.P(j,1:Nin)*x(Co+in) + Xr.P(j,Nin+1:Nin+Nout)*x(Cv+out);
        eqCount = eqCount + 1;
        j=j+1;
    else    % connecting segment  &  bifurcation
        if nnz(sg)==2
            F{eqCount} = @(x) x(Co+lsg) - x(Co+sg(1)) - x(Co+sg(2));
        elseif nnz(sg)==3
            F{eqCount} = @(x) x(Co+lsg) - x(Co+sg(1)) - x(Co+sg(2)) - x(Co+sg(3));
        elseif isempty(sg)
            F{eqCount} = @(x) x(Co+lsg(1)) + x(Co+lsg(2));
        else
            F{eqCount} = @(x) x(Co+lsg) - x(Co+sg);
        end
        eqCount = eqCount + 1;
    end
end
for i = 1 : vnode.n
    lsg = vsegment.num(vsegment.end==i);
    sg = vsegment.num(vsegment.init==i); 
    if vnode.isroot(i)==1
        F{eqCount} = @(x) x(node.n+i) - Pout;
        eqCount = eqCount + 1;
    elseif vnode.isterminal(i)==1
        F{eqCount} = @(x) x(node.n+i) - x(xend) + prm.mu/prm.gamma*x(Cv+lsg) - Xr.P(j,1:Nin)*x(Co+in) + Xr.P(j,Nin+1:Nin+Nout)*x(Cv+out);
        eqCount = eqCount + 1;
        j=j+1;
    elseif vnode.isroot(i)==0
        if nnz(sg)==2
            F{eqCount} = @(x) x(Cv+lsg) - x(Cv+sg(1)) - x(Cv+sg(2));
        elseif nnz(sg)==3
            F{eqCount} = @(x) x(Cv+lsg) - x(Cv+sg(1)) - x(Cv+sg(2)) - x(Cv+sg(3));
        elseif isempty(sg)
            F{eqCount} = @(x) x(Cv+lsg(1)) + x(Cv+lsg(2));
        else
            F{eqCount} = @(x) x(Cv+lsg) - x(Cv+sg);
        end
        eqCount = eqCount + 1;
    end
end
%% Mass conservation on system
lsg = Co+segment.num(segment.isroot);
sg = Cv+vsegment.num(vsegment.isroot);
F{eqCount} = @(x) x(lsg(1)) + x(lsg(2)) - x(sg(1)) - x(sg(2));

%% Solve
% guess = ones(2*Co-2,1);
options = optimoptions('fsolve','FunctionTolerance',1e-12,'MaxFunctionEvaluations',200000,'MaxIterations',100000);
[x,fval,exitflag,output,jacobian] = fsolve(@(y) cellfun(@(x) x(y), F), guess, options);
A=(@(y) cellfun(@(x) x(y), F));
norm(A(x))
% function angle3 = sudut(P0,P1,P2)
% n1 = (P2 - P0) / norm(P2 - P0);  % Normalized vectors
% n2 = (P1 - P0) / norm(P1 - P0);
% angle3 = atan2(norm(cross(n1, n2)), dot(n1, n2));
% end
end