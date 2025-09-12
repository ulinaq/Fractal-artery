function xlin = Frog_nonLinEq(prm,node,segment,vnode,vsegment,P)
Pin = prm.Pin; Qin = prm.Qin; Pout = prm.Pout;
Co = node.n + vnode.n; Cv = Co+segment.n;
eqCount = 1; j=1;
in = segment.num(segment.isroot); 
out = vsegment.num(vsegment.isroot);
Nin = nnz(in); Nout = nnz(out);
F = cell(Cv+vsegment.n+1);
b = zeros(Cv+vsegment.n+1,1);
%% Equation on segment (pressure drop)
% Artery 
for i = 1 : segment.n
    segint = segment.init(i);
    segend = segment.end(i);
    if node.junction(segment.init(i))   % pressure drop on bifurcation branch
        par = segment.num(segment.end==segment.init(i));
        cphi = cos(3*(segment.Gtheta(i)-segment.Gtheta(par))/4);
        F{eqCount} = @(x) x(segint) - x(segend) - 8*prm.mu*segment.len(i)*x(Co+i)/(pi*segment.rad(i)^4*prm.pixel^3)...
            - prm.Rho*x(Co+par)^2/(2*pi^2*segment.rad(par)^4*prm.pixel^4)...
            *(1+x(Co+i)^4*segment.rad(par)^4/(x(Co+par)^2*segment.rad(i)^4)...
            -2*segment.rad(par)^2*cphi*x(Co+i)/(x(Co+par)*segment.rad(i)^2));
    else
        F{eqCount} = @(x) x(segint) - x(segend) - 8*prm.mu*segment.len(i)*x(Co+sg(1))/(pi*segment.rad(i)^4*prm.pixel^3); % pressure drop
    end
    eqCount = eqCount + 1;
end
% Vein 
for i = 1 : vsegment.n
    segint = vsegment.init(i);
    segend = vsegment.end(i);
    F(eqCount,node.n+[segint,segend]) = [1,-1]; F(eqCount,Cv+i) = 8*prm.mu*vsegment.len(i)/(pi*vsegment.rad(i)^4*prm.pixel^3); % pressure drop
    eqCount = eqCount + 1;
end

%% Equation on node 
for i = 1 : node.n
    lsg = seg(segment.end==i);
    sg = seg(segment.init==i);
    if node.isroot(i)
        F(eqCount,i) = 1;   b(eqCount) = Pin; % Set pressure in root node as known
%         G(eqCount+1,Co+sg) = 1;   b(eqCount+1) = segment.rad(sg)^3*length(rot)*Qin/sum(segment.rad(rot).^3); % Set flowrate in the root segment
        eqCount = eqCount + 1;
    else if node.isterminal(i)==1       % Equation on terminal nodes
        F(eqCount,i) = 1;   F(eqCount,end) = -1;  F(eqCount,Co+in) = - P(j,1:Nin);   F(eqCount,Cv+out) = P(j,Nin+1:Nin+Nout);
        eqCount = eqCount + 1;
        j=j+1;
    else if node.isroot(i)==0    % connecting segment  &  bifurcation      
        F(eqCount,Co+lsg) = 1;      F(eqCount,Co+sg) = -1;
        eqCount = eqCount + 1;
    end
        end
    end
end

% Vein
% vrot= vsegment.num(vsegment.isroot);
% if nnz(vrot)==1 % Set outflow equal to inflow
%     G(eqCount,Cv+vrot) = 1;   b(eqCount) = Qin; % Set flowrate in the root segment
%     eqCount = eqCount + 1;
% else
%     for k=1:length(root)
%         G(eqCount+k-1,Cv+vrot(k)) = 1;     b(eqCount+k-1) = vsegment.rad(vrot(k))^3*length(rot)*Qin/sum(vsegment.rad(vrot).^3); % Set flowrate in the root segment
%     end
% 
%     eqCount = eqCount + length(vrot);
% end 
for i = 1 : vnode.n
    lsg = seg(vsegment.end==i);
    sg = seg(vsegment.init==i); 
    if vnode.isroot(i)==1
        F(eqCount,node.n+i) = 1;   b(eqCount) = Pout; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif vnode.isterminal(i)==1
        F(eqCount,node.n+i) = 1;   F(eqCount,end) = -1;  F(eqCount,Co+in) = - P(j,1:Nin);   F(eqCount,Cv+out) = P(j,Nin+1:Nin+Nout);
        eqCount = eqCount + 1;
        j=j+1;
    elseif vnode.isroot(i)==0
        F(eqCount,Cv+lsg) = 1;      F(eqCount,Cv+sg) = -1;   
        eqCount = eqCount + 1;
    end
end
%% Mass conservation on system
F(eqCount,Co+segment.num(segment.isroot))=1; F(eqCount,Cv+vsegment.num(vsegment.isroot))=-1; 

xlin = F\b;
end
