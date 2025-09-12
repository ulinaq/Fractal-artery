% function xlin = Frog_LinEq3(prm,Grid,node,segment,vnode,vsegment)
Pin = prm.Pin; Pout = prm.Pout;
Co = node.n + vnode.n; Cv = Co+segment.n;
eqCount = 1; j=1;
in = segment.num(segment.isterminal); 
out = vsegment.num(vsegment.isterminal);
Nin = nnz(in); Nout = nnz(out);
G = zeros(Cv+vsegment.n);
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
    G(eqCount,Cv+i) = -8*prm.mu*vsegment.len(i)/(pi*vsegment.rad(i)^4); % pressure drop
    eqCount = eqCount + 1;
end

%% Equation on node 
for i = 1 : node.n
    lsg = segment.num(segment.end==i);          
    sg = segment.num(segment.init==i);
    if node.isroot(i)
        G(eqCount,i) = 1;   b(eqCount) = Pin; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif node.isterminal(i)==1       % Equation on terminal nodes
        G(eqCount,i) = 1;      
        G(eqCount,Co+lsg) = -prm.mu/prm.gamma;   % pressure drop estimation on terminal nodes
        eqCount = eqCount + 1;
    elseif node.isroot(i)==0    % connecting segment  &  bifurcation      
        G(eqCount,Co+lsg) = 1;      G(eqCount,Co+sg) = -1;
        eqCount = eqCount + 1;
    end
end

% Vein
for i = 1 : vnode.n
    lsg = vsegment.num(vsegment.end==i);
    sg = vsegment.num(vsegment.init==i); 
    if vnode.isroot(i)
        G(eqCount,node.n+i) = 1;   b(eqCount) = Pout; % Set pressure in root node as known
        eqCount = eqCount + 1;
    elseif vnode.isterminal(i)==1
        G(eqCount,node.n+i) = 1;
        G(eqCount,Cv+lsg) = -prm.mu/prm.gamma;   % pressure drop estimation on terminal nodes
        eqCount = eqCount + 1;
    elseif vnode.isroot(i)==0
        G(eqCount,Cv+lsg) = 1;      G(eqCount,Cv+sg) = -1;   
        eqCount = eqCount + 1;
    end
end
%% Mass conservation on system
G(eqCount,Co+segment.num(segment.isroot))=1; G(eqCount,Cv+vsegment.num(vsegment.isroot))=1; 
% Solve
% xlin = G\b;
%% Re-arrange matrix G,b
Na = node.num(~node.isterminal); Nv = vnode.num(~vnode.isterminal);
Ta = node.num(node.isterminal); Tv = vnode.num(vnode.isterminal);
Nsa = segment.num(~segment.isterminal); Nsv = vsegment.num(~vsegment.isterminal);
Tsa = segment.num(segment.isterminal); Tsv = vsegment.num(vsegment.isterminal);
%% rows
Gb = [G,b];
Gn = zeros(size(Gb));
Gn(:,1:Co) = Gb(:,1:Co);        % Lokasi node tetap
Gn(:,Co+1:Co+length(Nsa)) = Gb(:,Co+Nsa);
Gn(:,Co+length(Nsa)+1:Co+length(Nsa)+length(Nsv)) = Gb(:,Cv+Nsv);
Gn(:,Co+length(Nsa)+length(Nsv)+1:Co+length(Nsa)+length(Nsv)+length(Tsa)) = Gb(:,Co+Tsa);
Gn(:,Co+length(Nsa)+length(Nsv)+length(Tsa)+1:Co+length(Nsa)+length(Nsv)+length(Tsa)+length(Tsv)) = Gb(:,Cv+Tsv);
Gn(:,end) = Gb(:,end);
% Columns
Gb = Gn; Gn = zeros(size(Gb));
Gn(1:segment.n+vsegment.n,:) = Gb(1:segment.n+vsegment.n,:);
Gn(segment.n+vsegment.n+1:segment.n+vsegment.n+length(Na),:) = Gb(segment.n+vsegment.n+Na,:);
Gn(segment.n+vsegment.n+length(Na)+1:segment.n+vsegment.n+length(Na)+length(Nv),:) = Gb(segment.n+vsegment.n+node.n+Nv,:);
Gn(segment.n+vsegment.n+length(Na)+length(Nv)+1,:) = Gb(end,:);
Gn(segment.n+vsegment.n+length(Na)+length(Nv)+2:1+segment.n+vsegment.n+length(Na)+length(Nv)+length(Ta),:) = Gb(segment.n+vsegment.n+Ta,:);
Gn(segment.n+vsegment.n+length(Na)+length(Nv)+length(Ta)+2:1+segment.n+vsegment.n+length(Na)+length(Nv)+length(Ta)+length(Tv),:) = Gb(segment.n+vsegment.n+node.n+Tv,:);
bn = Gn(:,end);
Gn = Gn(:,1:end-1);
% Block A_T-T
AT = Gn(end-185:end,end-185:end);
%% Solve 
[Xr,Grid,node,vnode] = FO_Capillary3(node,vnode,Grid);
Gn(end-185:end,end-185:end) = AT - Xr.ATX*Xr.C;
x = Gn\bn;
P = - Xr.C*x(end-185:end);
% end
% Re-arrange the solution
node.P = x(1:node.n);               vnode.P = x(node.n+1:Co);
segment.Q = zeros(segment.n,1);     vsegment.Q = zeros(vsegment.n,1);
segment.Q(Nsa) = x(Co+1:Co+length(Nsa));   
segment.Q(Tsa) = x(Co+length(Nsa)+length(Nsv)+1:Co+length(Nsa)+length(Nsv)+length(Tsa));
vsegment.Q(Nsv) = x(Co+length(Nsa)+1:Co+length(Nsa)+length(Nsv));   
vsegment.Q(Tsv) = x(Co+length(Nsa)+length(Nsv)+length(Tsa)+1:Co+length(Nsa)+length(Nsv)+length(Tsa)+length(Tsv));
