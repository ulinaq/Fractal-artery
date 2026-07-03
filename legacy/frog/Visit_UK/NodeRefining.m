% function [state,Grid] = FS_Capillary(node,vnode,segment,vsegment,Grid)
a = [node.input;vnode.output];
RepsA = 30; RepsV = 30;   % Epsilon radii for mollifier around terminal nodes
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz);
ain = node.num(node.isterminal);
vout = vnode.num(vnode.isterminal);
rmin = min(segment.rad);
for i=1:length(node.input)
    % Create circle region for spreading blood region
    Center = node.input(i);
    Rc = RepsA*(segment.rad(segment.end==ain(i))/rmin)^2;
    [bw1,bw2] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    Srdg(i) = any(ismember(vnode.output,Grid.num(bw1)));
    if ~Srdg(i)
        segment.rad(segment.end==ain(i))=1e-4;
    end
end

rmin = min(vsegment.rad);
for j=1:length(vnode.output)
    Center = vnode.output(j);
    Rc = RepsV*vsegment.rad(vsegment.end==vout(j))/rmin;
    [bw1,bw2] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    Srd(j) = any(ismember(node.input,Grid.num(bw1)));
    if ~Srd(j)
        vsegment.rad(vsegment.end==vout(j))=1e-4;
    end

end

function [bw1,bw2] = mollifier(xcell,ycell,zcell,Reps,Grid,Center)
        bw1=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Reps^2;
        bw2=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=8*Reps^2;
end
% end