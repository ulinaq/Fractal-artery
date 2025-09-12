function [node,segment] = Bif_angle(node,segment)
% Compute global angle relative to parent segment
for i = 1:segment.n
    par = segment.num(segment.end==segment.init(i)); % parent segment
    if segment.isroot(i)
        Cen = node.avc(segment.init(i));
        Cenp = Cen; Cenp(2)=Cenp(2)+1;
        segment.Gtheta(i)= sudut(Cen,Cenp,node.avc(segment.end(i)));
        Cen = segment.c{i}(end);
        Cenp = Cen; Cenp(2)=Cenp(2)+1;
        segment.Ltheta(i)= sudut(Cen,Cenp,segment.c{i}(end-2));
    else
        segment.Gtheta(i)= sudut(node.avc(segment.init(i)),node.avc(par),node.avc(segment.end(i)));
        segment.Ltheta(i)= sudut(segment.c{i}(end),segment.c{par}(2),segment.c{i}(end-2));
    end  
end




% Compute the angle in bifurcation relative to vertical line
function angle = sudut(P0,P1,P2)
% P0 = [x0, y0, z0]; middle
% P1 = [x1, y1, z0]; parent vessel
% P2 = [x2, y2, z0]; child vessel
n1 = (P2 - P0) / norm(P2 - P0);  % Normalized vectors
n2 = (P1 - P0) / norm(P1 - Po);
angle = atan2(norm(cross(n1, n2)), dot(n1, n2));  % Stable
end