function [node,segment,x] = FrognonlinEq(node,prm,segment,vnode,vsegment,Xr)
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
    if node.junction(segint)
        par = segment.num(segment.end==segint);
        cphi = cos(3*(segment.Gtheta(i)-segment.Gtheta(par))/4);
        F{eqCount} = @(x) x(segint) - x(segend) - 8*prm.mu*segment.len(i)*x(Co+i)/(pi*segment.rad(i)^4)...
            *(1-3*(x(segint)/2 + x(segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.ha*prm.E))...
            -  prm.Rho*x(Co+par)^2/(2*pi^2*segment.rad(par)^4)...
            *(1+x(Co+i)^2*segment.rad(par)^4/(x(Co+par)^2*segment.rad(i)^4)...
            -2*x(Co+i)*segment.rad(par)^2*cphi*x(Co+i)/(x(Co+par)*segment.rad(i)^2));
        eqCount = eqCount + 1; k=k+1;
    else % no junction
        F{eqCount} = @(x) x(segint) - x(segend) - 8*prm.mu*segment.len(i)*x(Co+i)/(pi*segment.rad(i)^4)...
            *(1-3*(x(segint)/2 + x(segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.ha*prm.E));
        eqCount = eqCount + 1; k=k+1;
    end
end
% Vein 
for i = 1 : vsegment.n
    segint = vsegment.init(i);
    segend = vsegment.end(i);
    if vnode.junction(segint)
        par = vsegment.num(vsegment.end==segint);
        cphi = cos(3*(vsegment.Gtheta(i)-vsegment.Gtheta(par))/4);
        F{eqCount} = @(x) x(node.n+segint) - x(node.n+segend) + 8*prm.mu*vsegment.len(i)*x(Cv+i)/(pi*vsegment.rad(i)^4)...
            *(1-3*(x(node.n+segint)/2 + x(node.n+segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.hv*prm.E))...
            + prm.Rho*x(Cv+par)^2/(2*pi^2*vsegment.rad(par)^4)...
            *(1+x(Cv+i)^2*vsegment.rad(par)^4/(x(Cv+par)^2*vsegment.rad(i)^4)...
            -2*x(Cv+i)*vsegment.rad(par)^2*cphi*x(Cv+i)/(x(Cv+par)*vsegment.rad(i)^2));
        eqCount = eqCount + 1; k=k+1;
    else % no junction
        F{eqCount} = @(x) x(node.n+segint) - x(node.n+segend) + 8*prm.mu*vsegment.len(i)*x(Cv+i)/(pi*vsegment.rad(i)^4)...
            *(1-3*(x(node.n+segint)/2 + x(node.n+segend)/2 - x(xend) - Xr.Pext(k,1:Nin)*x(Co+in)...
            + Xr.Pext(k,Nin+1:Nin+Nout)*x(Cv+out))/(prm.hv*prm.E));
        eqCount = eqCount + 1; k=k+1;
    end
end
%% Equation on node 
for i = 1 : node.n
    lsg = segment.num(segment.end==i);
    sg = segment.num(segment.init==i);
    if node.isroot(i)
        F{eqCount} = @(x) x(i) - Pin; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif node.isterminal(i)       % Equation on terminal nodes
        if node.clog(i) % the terminal node is blocked (no connection with capillary domain)
            F{eqCount} = @(x) x(Co+lsg);
        else
            F{eqCount} = @(x) x(i) - x(xend) - prm.mu/prm.gamma*x(Co+lsg) - Xr.P(j,1:Nin)*x(Co+in) + Xr.P(j,Nin+1:Nin+Nout)*x(Cv+out);
        end
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
        if node.clog(i) % the terminal node is blocked (no connection with capillary domain)
            F{eqCount} = @(x) x(Cv+lsg);
        else
            F{eqCount} = @(x) x(node.n+i) - x(xend) + prm.mu/prm.gamma*x(Cv+lsg) - Xr.P(j,1:Nin)*x(Co+in) + Xr.P(j,Nin+1:Nin+Nout)*x(Cv+out);
        end
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
options = optimoptions('fsolve','FunctionTolerance',1e-8,'MaxFunctionEvaluations',200000,'MaxIterations',100000);
[x,fval,exitflag,output,jacobian] = fsolve(@(y) cellfun(@(x) x(y), F), guess, options);
end