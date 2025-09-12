function xlin = Frog_LinEq2(prm,Grid,node,segment,vnode,vsegment,P)
Pin = prm.Pin; Pout = prm.Pout;
eqCount = 1; 
in = segment.num(segment.isterminal); 
out = vsegment.num(vsegment.isterminal);
Nin = nnz(in); Nout = nnz(out);
Na = node.num(~node.isterminal); Nv = vnode.num(~vnode.isterminal);
Ta = node.num(node.isterminal); Tv = vnode.num(vnode.isterminal);
Nsa = segment.num(~segment.isterminal); Nsv = vsegment.num(~vsegment.isterminal);
Tsa = segment.num(segment.isterminal); Tsv = vsegment.num(vsegment.isterminal);
Co = numel(Na)+numel(Nv); Cv = Co+numel(Nsa);
G = zeros(Cv+vsegment.n+1);
b = zeros(Cv+vsegment.n+1,1);
%% Equation on segment (pressure drop) Internal node
% Artery (Block matrix G_Na-Na and G_Na-Nsa)
for i = 1 : length(Nsa)
    segint = node.num(Na==segment.init(i));
    segend = node.num(Na==segment.end(i));
    j = Nsa(i); 
    G(eqCount,[segint,segend]) = [1,-1]; 
    G(eqCount,Co+i) = -8*prm.mu*segment.len(j)/(pi*segment.rad(j)^4); % pressure drop
    eqCount = eqCount + 1;
end
% Vein (Block matrix G_Nv-Nv and G_Nv-Nsv)
for i = 1 : length(Nsv)
    segint = vnode.num(Nv==vsegment.init(i));
    segend = vnode.num(Nv==vsegment.end(i));
    j = Nsv(i); 
    G(eqCount,numel(Na)+[segint,segend]) = [1,-1]; 
    G(eqCount,Cv+i) = -8*prm.mu*segment.len(j)/(pi*segment.rad(j)^4); % pressure drop
    eqCount = eqCount + 1;
end

%% Equation on node 
for i = 1 : length(Na)
    if node.isroot(Na(i))
        G(eqCount,i) = 1;   b(eqCount) = Pin; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif       % connecting segment  &  bifurcation      
        lsg = segment.num(Nsa==segment.num(segment.end==Na(i)));    %parent
        sg = segment.num(segment.init==Na(i));    %childs
        G(eqCount,Co+lsg) = 1;      G(eqCount,Co+sg) = -1;
        eqCount = eqCount + 1;
    end
end

% Vein
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
eqCount = eqCount + 1;

%% Equation on segment (pressure drop) terminal node
% Artery 
for i = 1 : segment.n
    if ~segment.isterminal(i)
    segint = segment.init(i);
    segend = segment.end(i);
    G(eqCount,[segint,segend]) = [1,-1]; 
    G(eqCount,Co+i) = -8*prm.mu*segment.len(i)/(pi*segment.rad(i)^4); % pressure drop
    eqCount = eqCount + 1;
    end
end
% Vein 
for i = 1 : vsegment.n
    if ~vsegment.isterminal(i)
    segint = vsegment.init(i);
    segend = vsegment.end(i);
    G(eqCount,node.n+[segint,segend]) = [1,-1]; 
    G(eqCount,Cv+i) = 8*prm.mu*vsegment.len(i)/(pi*vsegment.rad(i)^4); % pressure drop
    eqCount = eqCount + 1;
    end
end


xlin = G\b;
end
