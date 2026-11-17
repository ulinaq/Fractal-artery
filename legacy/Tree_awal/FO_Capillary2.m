function [state,Grid] = FO_Capillary2(node,vnode,Grid)

a = [node.input;vnode.output];
zn = [];
Grid.src.cell = [];
Grid.src.rate = [];
% Create Block A_T-X
% ATX=zeros(length(a),2*Grid.N);   % ATX = AXT'
% Create Block A_T-X
AX=zeros(Grid.N,length(node.input));   % without source as default
AT=zeros(Grid.N,length(vnode.output));   % without source as default
zn = 1*[0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989;0.0898876404494382;0.101123595505618;0.0898876404494382;0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989];
for i=1:length(node.input)
    % Create circle region for spreading epsilon
    area= [a(i)-2*Grid.Nx, a(i)-Grid.Nx-1, a(i)-Grid.Nx, a(i)-Grid.Nx+1, a(i)-2, a(i)-1, a(i), a(i)+1, a(i)+2, a(i)+Grid.Nx-1, a(i)+Grid.Nx, a(i)+Grid.Nx+1, a(i)+2*Grid.Nx]';
    AX(area,i)=zn;
    Grid.src.rate{i} = zn;
    Grid.src.cell = [Grid.src.cell, area];
end
zn = 1*[0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989;0.0898876404494382;0.101123595505618;0.0898876404494382;0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989];
for j=1:length(vnode.output)
    i=i+1;
    % Create circle region for spreading epsilon
    area= [a(i)-2*Grid.Nx, a(i)-Grid.Nx-1, a(i)-Grid.Nx, a(i)-Grid.Nx+1, a(i)-2, a(i)-1, a(i), a(i)+1, a(i)+2, a(i)+Grid.Nx-1, a(i)+Grid.Nx, a(i)+Grid.Nx+1, a(i)+2*Grid.Nx]';
    AT(area,j)=zn;
    Grid.src.rate{i} = zn;
    Grid.src.cell = [Grid.src.cell, area];
end
AXT = [AX,zeros(Grid.N,length(vnode.output));zeros(Grid.N,length(node.input)),AT];
state.ATX = sparse([1*AXT(:,1:length(node.input))';1*AXT(:,length(node.input)+1:end)']);
AXT = [AXT;zeros(1,length(a))];
% Solve system equation
pvec = Grid.Amat\AXT;
state.AXT = sparse(AXT);
state.C = pvec;
state.P = state.ATX*pvec;
end