function [state,Grid] = FD_Capillary(node,vnode,segment,vsegment,Grid)
a = [node.input;vnode.output];
RepsA = 5; RepsV = 5;   % Epsilon radii for mollifier around terminal nodes
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz);
ain = node.num(node.isterminal);
vout = vnode.num(vnode.isterminal);
Grid.src.cell = [];
Grid.src.rate = [];
% Create Block A_T-X
AX=zeros(Grid.N,length(node.input));   % without source as default
AT=zeros(Grid.N,length(vnode.output));   % without source as default
rmin = min(segment.rad);
breg = zeros(Grid.Nx,Grid.Ny);
for i=1:length(node.input)
    % Create circle region for spreading epsilon
    Center = node.input(i);
    Rc = RepsA*(segment.rad(segment.end==ain(i))/rmin)^2;
    [z,reg,bw] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    breg = breg | bw;
    AX(reg,i)=z(reg);
    Grid.src.rate{i} = z(reg);
    Grid.src.cell{i} = reg;
end
rmin = min(vsegment.rad);
for j=1:length(vnode.output)
    i=i+1;
    Center = vnode.output(j);
    Rc = RepsV*(vsegment.rad(vsegment.end==vout(j))/rmin)^2;
    [z,reg,bw] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    AT(reg,j)=z(reg);
    Grid.src.rate{i} = z(reg);
    Grid.src.cell{i} = reg;
end
AXT = [AX,zeros(Grid.N,length(vnode.output));zeros(Grid.N,length(node.input)),AT];
state.ATX = sparse([1*AXT(:,1:length(node.input))';1*AXT(:,length(node.input)+1:end)']);
AXT = [AXT;zeros(1,length(a))];
% Solve system equation
pvec = Grid.Amat\AXT;
state.AXT = sparse(AXT);
state.C = pvec;
state.P = state.ATX*pvec;


function [z,reg,bw] = mollifier(xcell,ycell,zcell,Reps,Grid,Center)
        area=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Reps^2;
        D=((xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2)/Reps^2;
        C = 1;
        z=C*exp(-1./(1-D).^8).*area; bw =area;
        z(~Grid.B|isnan(z)) = 0;
        area(~Grid.K) = false;
        z = z/sum(z(:));
        reg = Grid.num(area);
end
end