function [node,segment,x] = TreeSysEquation(guess,node,prm,segment,nodeCount,Pout,Qin)
Co = nodeCount;
eqCount = 1;
seg = 1:1:nodeCount-1;
j=1;
%% Create system of equation from artery
F{eqCount} = @(x) Pin - x(1) - 8*prm.mu*segment.len(1)*Qin/(pi*segment.rad(1)^4);
F{eqCount+1} = @(x) Qin - x(Co+1);
eqCount = eqCount + 2;
for i = 2 : nodeCount
    segend = segment.end==i;
    lsg = seg(segend);                      % parent segment
    ch = segment.end(segment.init==i);      % 2 nodes in bifurcation
    sg = seg(segment.init==i);              % 2 segments in bifurcation
    cphi = cos(3*(segment.theta(segment.init==i)-segment.theta(segend))/4);

    if node.junction(i)==1              % equation in bifurcation
            % equation
        F{eqCount} = @(x) x(i-1) - x(ch(1)-1) - 8*prm.mu*segment.len(sg(1))*x(Co+sg(1))/(pi*segment.rad(sg(1))^4)...
            - prm.Rho*x(Co+lsg)^2/(2*pi^2*segment.rad(segend)^4)...
            *(1+x(Co+sg(1))^4*segment.rad(segend)^4/(x(Co+lsg)^2*segment.rad(sg(1))^4)...
            -2*segment.rad(segend)^2*cphi(1)*x(Co+sg(1))/(x(Co+lsg)*segment.rad(sg(1))^2));
        F{eqCount+1} = @(x) x(i-1) - x(ch(2)-1) - 8*prm.mu*segment.len(sg(2))*x(Co+sg(2))/(pi*segment.rad(sg(2))^4)...
            - prm.Rho*x(Co+lsg)^2/(2*pi^2*segment.rad(segend)^4)...
            *(1+x(Co+sg(2))^4*segment.rad(segend)^4/(x(Co+lsg)^2*segment.rad(sg(2))^4)...
            -2*segment.rad(segend)^2*cphi(2)*x(Co+sg(2))/(x(Co+lsg)*segment.rad(sg(2))^2));
        F{eqCount+2} = @(x) x(Co+lsg) - x(Co+sg(1)) - x(Co+sg(2));
        eqCount = eqCount + 3;
        else
            F{eqCount} = @(x) x(i-1) - x(Co) - prm.Rho*x(Co+sg)^2/(2*pi^2*segment.rad(segment.end==i)^4);
            eqCount = eqCount + 1;
        end
end

% Solve
options = optimoptions('fsolve','FunctionTolerance',1e-8,'MaxFunctionEvaluations',200000,'MaxIterations',100000);
[x,fval,exitflag,output,jacobian] = fsolve(@(y) cellfun(@(x) x(y), F), guess, options);

% Put the solution to the node
node.FlowRate(1) = Qin;
node.FlowRate(2) = Qin;
odd=2;
even=3;
for i=1:nodeCount
    if bitget(i,1)
        node.Pressure(odd) = x(i);
        odd = odd+1;
    else
        node.FlowRate(even) = x(i);
        even = even+1;
    end
end
node.Pressure(1) = node.Pressure(2) + 8*prm.mu*segment.len(1)*node.FlowRate(1)/(pi*segment.rad(1)^4);

end