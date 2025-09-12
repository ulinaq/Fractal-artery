% function [state,Grid] = FS_Capillary(node,vnode,segment,vsegment,Grid)
a = [node.input;vnode.output];
RepsA = 10; RepsV = 20;   % Epsilon radii for mollifier around terminal nodes
Grid.num = [1:1:Grid.N]';
[ycell,xcell,zcell]= meshgrid(1:Grid.Ny,1:Grid.Nx,1:Grid.Nz);
ain = node.num(node.isterminal);
vout = vnode.num(vnode.isterminal);
rmin = min(segment.rad);
breg = zeros(Grid.Size);
breg2 = zeros(Grid.Size);
for i=1:length(node.input)
    % Create circle region for spreading epsilon
    Center = node.input(i);
    Rc = RepsA*(segment.rad(segment.end==ain(i))/rmin)^2;
    [bw1,bw2] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
    breg = breg | bw1;
    breg2 = breg2 | bw2;
end
imshow(breg2);
Dd = bwdist(breg);
imshow(Dd,[]);
% Dd = -Dd;
DL = watershed(Dd); 
imshow(DL);
% DL(~breg)=0;
bgm = DL == 0;
bgm = bwmorph(bgm,'thicken');
% bgm = bgm | breg;
imshow(bgm)
brr = breg2 - bgm;
imshow(brr)
brr(~Grid.B)=0;
% rmin = min(vsegment.rad);
% breg2 = zeros(Grid.Size);
% for j=1:length(vnode.output)
%     i=i+1;
%     Center = vnode.output(j);
%     Rc = RepsV*vsegment.rad(vsegment.end==vout(j))/rmin;
%     [z,reg,bw] = mollifier(xcell,ycell,zcell,Rc,Grid,Center); 
%     breg2 = breg2 | bw;
%     AX(reg,i)=z(reg);
%     Grid.src.rate{i} = z(reg);
%     Grid.src.cell{i} = reg;
% end

function [bw1,bw2] = mollifier(xcell,ycell,zcell,Reps,Grid,Center)
        bw1=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=Reps^2;
        bw2=(xcell-xcell(Center)).^2+(ycell-ycell(Center)).^2<=8*Reps^2;
end
% end