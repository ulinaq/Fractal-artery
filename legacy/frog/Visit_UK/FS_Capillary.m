function [state,Grid] = FS_Capillary(node,vnode,segment,vsegment,Grid)
a = [node.input;vnode.output];
RepsA = 100; RepsV = 100;   % Epsilon radii for mollifier around terminal nodes
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz);
ain = node.num(node.isterminal);
vout = vnode.num(vnode.isterminal);
Grid.src.cell = [];
Grid.src.rate = [];
% Create Block A_T-X
AX=zeros(Grid.N,length(a));   % without source as default
rmin = min(segment.rad);
for i=1:length(node.input)
    % Create circle region for spreading epsilon
    Center = node.input(i);
    Rc = RepsA*segment.rad(segment.end==ain(i))/rmin;
    [z,reg] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    AX(reg,i)=z(reg);
    Grid.src.rate{i} = z(reg);
    Grid.src.cell{i} = reg;
end
rmin = min(vsegment.rad);
for j=1:length(vnode.output)
    i=i+1;
    Center = vnode.output(j);
    Rc = RepsV*vsegment.rad(vsegment.end==vout(j))/rmin;
    [z,reg] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    AX(reg,i)=z(reg);
    Grid.src.rate{i} = z(reg);
    Grid.src.cell{i} = reg;
end
state.ATX = sparse([1*AX(:,1:length(node.input))';1*AX(:,length(node.input)+1:end)']);
AXT = [AX;zeros(1,length(a))];
% Solve system equation
pvec = Grid.Amat\AXT;
state.AXT = sparse(AXT);
state.C = pvec;
state.P = state.ATX*pvec;

function [z,reg] = mollifier(xcell,ycell,zcell,Reps,Grid,Center)
        area=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<Reps^2;
        D=((xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2)/Reps^2;
        C = 1;
        z=C*exp(-1./(1-D)).*area;
        z(~Grid.B|isnan(z)) = 0;
        area(~Grid.B) = false;
        z = z/sum(z(:));
        reg = Grid.num(area);
end
end