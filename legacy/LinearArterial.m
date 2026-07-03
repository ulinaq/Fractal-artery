function xlin = LinearArterial(prm,node,segment,Pin,Qin,nodeCount)
eqCount = 1;
seg = 1:1:nodeCount-1;
G=zeros(2*nodeCount-3);
b=zeros(2*nodeCount-3,1);
b(1) = Pin;
%% Create system of equation and Jacobian matrix
for i = 2 : nodeCount
    lsg = seg(segment.end==i);              % parent segment
    ch = segment.end(segment.init==i);      % 2 nodes in bifurcation
    sg = seg(segment.init==i);               % 2 segments in bifurcation
    cphi = cos(3*(segment.theta(segment.init==i)-segment.theta(segment.end==i))/4);

    if node.junction(i)==1              % equation in bifurcation
        % equation
        G(eqCount,2*i-3) = 1;   G(eqCount,2*ch(1)-3) = -1;  G(eqCount,2*sg(1)-2) = -8*prm.mu*segment.len(sg(1))/(pi*segment.rad(sg(1))^4);
        G(eqCount+1,2*i-3) = 1; G(eqCount+1,2*ch(2)-3) = -1;  G(eqCount+1,2*sg(2)-2) = -8*prm.mu*segment.len(sg(2))/(pi*segment.rad(sg(2))^4);
        if i==2 % first bifurcation
        G(eqCount+2,2*sg-2) = 1; 
        b(eqCount+2) = Qin;
        else
            G(eqCount+2,2*lsg-2) = 1;  G(eqCount+2,2*sg-2) = -1;
        end
        eqCount = eqCount + 3;
    else % final node
            G(eqCount,2*i-3) = 1;
            b(eqCount) = Pout;
%             - prm.Rho*x(2*lsg-1)^2/(2*pi^2*segment.rad(segment.end==i)^4);
            eqCount = eqCount + 1;
        end
    end

xlin = G\b;

end
