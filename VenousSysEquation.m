function [vnode,vsegment,x] = VenousSysEquation(vnode,prm,vsegment,vnodeCount,Pout,Qin)
%% Linear
eqCount = 1;
seg = 1:1:vnodeCount-1;
G=zeros(2*vnodeCount-3);
b=zeros(2*vnodeCount-3,1);
%% Create system of equation
for i = 2 : vnodeCount
    lsg = seg(vsegment.end==i);              % parent segment
    ch = vsegment.end(vsegment.init==i);      % 2 nodes in bifurcation
    sg = seg(vsegment.init==i);              % 2 segments in bifurcation
    cphi = cos(3*(vsegment.theta(vsegment.init==i)-vsegment.theta(vsegment.end==i))/4);

    if vnode.junction(i)==1              % equation in bifurcation
        % equation
        G(eqCount,2*i-3) = 1;   G(eqCount,2*ch(1)-3) = -1;  G(eqCount,2*sg(1)-2) = 8*prm.mu*vsegment.len(sg(1))/(pi*vsegment.rad(sg(1))^4);
        G(eqCount+1,2*i-3) = 1; G(eqCount+1,2*ch(2)-3) = -1;  G(eqCount+1,2*sg(2)-2) = 8*prm.mu*vsegment.len(sg(2))/(pi*vsegment.rad(sg(2))^4);
        if i==2
        G(eqCount+2,2*sg-2) = 1; 
        b(eqCount+2) = Qin;
        else
            G(eqCount+2,2*lsg-2) = 1;  G(eqCount+2,2*sg-2) = -1;
        end
        eqCount = eqCount + 3;
        else
            G(eqCount,2*i-3) = 1;
            b(eqCount) = Pout;
%             - prm.Rho*x(2*lsg-1)^2/(2*pi^2*segment.rad(segment.end==i)^4);
            eqCount = eqCount + 1;
        end
    end

xlin = G\b;

%% Create nonlinear system of equation
eqCount = 1;
seg = 1:1:vnodeCount-1;

for i = 2 : vnodeCount
    lsg = seg(vsegment.end==i);              % parent segment
    ch = vsegment.end(vsegment.init==i);      % 2 nodes in bifurcation
    sg = seg(vsegment.init==i);              % 2 segments in bifurcation
    cphi = cos(3*(vsegment.theta(vsegment.init==i)-vsegment.theta(vsegment.end==i))/4);

    if vnode.junction(i)==1              % equation in bifurcation
            % equation
        F{eqCount} = @(x) x(2*i-3) - x(2*ch(1)-3) + 8*prm.mu*vsegment.len(sg(1))*x(2*sg(1)-2)/(pi*vsegment.rad(sg(1))^4)...
            + prm.Rho*Qin^2/(2*pi^2*vsegment.len(vsegment.end==i)^4)*x(2*sg(1)-2)...
            *(1+(x(2*i-2)*vsegment.len(vsegment.end==i)^4/(Qin^2*vsegment.len(sg(1))^4))...
            -2*vsegment.len(vsegment.end==i)^2*cphi(1)/(Qin*vsegment.len(sg(1))^2));
        F{eqCount+1} = @(x) x(2*i-3) - x(2*ch(2)-3) + 8*prm.mu*vsegment.len(sg(2))*x(2*sg(2)-2)/(pi*vsegment.rad(sg(2))^4)...
            + prm.Rho*Qin^2/(2*pi^2*vsegment.len(vsegment.end==i)^4)*x(2*sg(2)-2)...
            *(1+(x(2*i-2)*vsegment.len(vsegment.end==i)^4/(Qin^2*vsegment.len(sg(2))^4))...
            -2*vsegment.len(vsegment.end==i)^2*cphi(1)/(Qin*vsegment.len(sg(2))^2));
        if i==2
            F{eqCount+2} = @(x) Qin - x(2*sg(2)-2) - x(2*sg(1)-2);
        else
            F{eqCount+2} = @(x) x(2*lsg-2) - x(2*sg(2)-2) - x(2*sg(1)-2);
        end
        eqCount = eqCount + 3;
        else
            F{eqCount} = @(x) x(2*i-3) - Pout + prm.Rho*x(2*lsg-2)^2/(2*pi^2*vsegment.rad(vsegment.end==i)^4);
            eqCount = eqCount + 1;
        end
    end

% Solve
options = optimoptions('fsolve','FunctionTolerance',1e-8,'MaxFunctionEvaluations',200000,'MaxIterations',100000);
[x,fval,exitflag,output,jacobian] = fsolve(@(y) cellfun(@(x) x(y), F), xlin, options);

% Put the solution to the node
vnode.FlowRate(1) = Qin;
vnode.FlowRate(2) = Qin;
odd=2;
even=3;
for i=1:2*vnodeCount-3
    if bitget(i,1)
        vnode.Pressure(odd) = x(i);
        odd = odd+1;
    else
        vnode.FlowRate(even) = x(i);
        even = even+1;
    end
end
vnode.Pressure(1) = vnode.Pressure(2) - 8*prm.mu*vsegment.len(1)*vnode.FlowRate(1)/(pi*vsegment.rad(1)^4);

end