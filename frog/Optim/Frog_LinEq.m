function xlin = Frog_LinEq(prm,node,segment,vnode,vsegment,P)
Pin = prm.Pin; Pout = prm.Pout;
Co = node.n + vnode.n; Cv = Co+segment.n;
eqCount = 1; j=1;
in = segment.num(segment.isterminal); 
out = vsegment.num(vsegment.isterminal);
Nin = nnz(in); Nout = nnz(out);
G = zeros(Cv+vsegment.n+1);
b = zeros(Cv+vsegment.n+1,1);
%% Equation on segment (pressure drop) Internal node
% Artery 
for i = 1 : segment.n
    segint = segment.init(i);
    segend = segment.end(i);
    G(eqCount,[segint,segend]) = [1,-1]; 
    G(eqCount,Co+i) = -8*prm.mu*segment.len(i)/(pi*segment.rad(i)^4); % pressure drop
    eqCount = eqCount + 1;
end
% Vein 
for i = 1 : vsegment.n
    segint = vsegment.init(i);
    segend = vsegment.end(i);
    G(eqCount,node.n+[segint,segend]) = [1,-1]; 
    G(eqCount,Cv+i) = 8*prm.mu*vsegment.len(i)/(pi*vsegment.rad(i)^4); % pressure drop
    eqCount = eqCount + 1;
end

%% Equation on node 
for i = 1 : node.n
    lsg = segment.num(segment.end==i);
    sg = segment.num(segment.init==i);
    if node.isroot(i)
        G(eqCount,i) = 1;   b(eqCount) = Pin; % Set pressure in root node as known
%         G(eqCount+1,Co+sg) = 1;   b(eqCount+1) = segment.rad(sg)^3*length(rot)*Qin/sum(segment.rad(rot).^3); % Set flowrate in the root segment
        eqCount = eqCount + 1;
    elseif node.isterminal(i)==1       % Equation on terminal nodes
        G(eqCount,i) = 1;   G(eqCount,end) = -1;  G(eqCount,Co+in) = - P(j,1:Nin);   
        G(eqCount,Cv+out) = P(j,Nin+1:Nin+Nout);    
        G(eqCount,Co+lsg) = G(eqCount,Co+lsg) - prm.mu/prm.gamma;   % pressure drop estimation on terminal nodes
        eqCount = eqCount + 1;
        j=j+1;
    elseif node.isroot(i)==0    % connecting segment  &  bifurcation      
        G(eqCount,Co+lsg) = 1;      G(eqCount,Co+sg) = -1;
        eqCount = eqCount + 1;
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
    lsg = vsegment.num(vsegment.end==i);
    sg = vsegment.num(vsegment.init==i); 
    if vnode.isroot(i)==1
        G(eqCount,node.n+i) = 1;   b(eqCount) = Pout; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif vnode.isterminal(i)==1
        G(eqCount,node.n+i) = 1;   G(eqCount,end) = -1;  G(eqCount,Co+in) = - P(j,1:Nin);   
        G(eqCount,Cv+out) = P(j,Nin+1:Nin+Nout);
        G(eqCount,Cv+lsg) = G(eqCount,Cv+lsg) + prm.mu/prm.gamma;   % pressure drop estimation on terminal nodes
        eqCount = eqCount + 1;
        j=j+1;
    elseif vnode.isroot(i)==0
        G(eqCount,Cv+lsg) = 1;      G(eqCount,Cv+sg) = -1;   
        eqCount = eqCount + 1;
    end
end
%% Mass conservation on system
G(eqCount,Co+segment.num(segment.isroot))=1; G(eqCount,Cv+vsegment.num(vsegment.isroot))=-1; 

xlin = G\b;
end
