function [state,Grid,node,vnode] = FO_Capillary(node,vnode,Grid)

a = [node.input;vnode.output];
zn = [];
Grid.src.cell = [];
Grid.src.rate = [];
cell = [];
q=zeros(Grid.N+1,length(a));   % without source as default
for i=1:length(a)
    % Create circle region for spreading epsilon
    area= [a(i)-2*Grid.Nx, a(i)-Grid.Nx-1, a(i)-Grid.Nx, a(i)-Grid.Nx+1, a(i)-2, a(i)-1, a(i), a(i)+1, a(i)+2, a(i)+Grid.Nx-1, a(i)+Grid.Nx, a(i)+Grid.Nx+1, a(i)+2*Grid.Nx]';
    zn = [0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989;0.0898876404494382;0.101123595505618;0.0898876404494382;0.0561797752808989;0.0786516853932584;0.0898876404494382;0.0786516853932584;0.0561797752808989];
    q(area,i)=zn;
    Grid.src.rate{i} = zn;
    Grid.src.cell{i} = [area];
    cell = [cell; area];
end
% Solve system equation
state.pvec = Grid.Amat\q;
nz = numel(zn);
for i=1:length(a)
    state.P(i,:)=zn'*state.pvec(cell(nz*(i-1)+1:nz*i),:);
end
end

