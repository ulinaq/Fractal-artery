function [segment] = Bif_angle(node,segment)
for i = 1:segment.n
    segment.Gtheta(i)= angle(node.avc(segment.init(i),:),node.avc(segment.end(i),:));
    if isempty(segment.c{i})
        segment.Ltheta(i)=segment.Gtheta(i);
    elseif min(size(segment.c{i}))>4
        segment.Ltheta(i)= angle(segment.c{i}(end,:),segment.c{i}(end-3,:));
    elseif min(size(segment.c{i}))>2
        segment.Ltheta(i)= angle(segment.c{i}(end,:),segment.c{i}(end-1,:));
    else
        segment.Ltheta(i)= segment.Gtheta(i);
    end
     
end
% angle of segment respect to x-axis (2D)
function ksi = angle(Pinit,Pend) 
    a=sqrt((Pinit(1)-Pend(1))^2+(Pinit(2)-Pend(2))^2);   % scale factor
    b=1;
    c=sqrt((Pinit(1)+1-Pend(1))^2+(Pinit(2)-Pend(2))^2);
    alpha=acos((a^2+b^2-c^2)/(2*a*b));
    if Pend(2)>=Pinit(2)
        ksi=alpha;
    else
        ksi=-alpha; 
    end
end

% Compute global angle relative to parent segment
% for i = 1:segment.n
%     par = segment.num(segment.end==segment.init(i)); % parent segment
%     if segment.isroot(i)
%         Cen = node.avc(segment.init(i),:);
%         Cenp = Cen; Cenp(2)=Cenp(2)+1;
%         segment.Gtheta(i)= sudut(Cen,Cenp,node.avc(segment.end(i),:));
%         Cen = segment.c{i}(end,:);
%         Cenp = Cen; Cenp(2)=Cenp(2)+1;
%         segment.Ltheta(i)= sudut(Cen,Cenp,segment.c{i}(end-1,:));
%     else
%         segment.Gtheta(i)= sudut(node.avc(segment.init(i),:),node.avc(segment.init(par),:),node.avc(segment.end(i),:));
%         if isempty(segment.c{i}) | isempty(segment.c{par}) | isempty(segment.c{i})
%             segment.Ltheta(i)=segment.Gtheta(i);
%         else if min(size(segment.c{i}))>2
%                 segment.Ltheta(i)= sudut(segment.c{i}(end,:),segment.c{par}(2,:),segment.c{i}(end-1,:));
%             else
%                 segment.Ltheta(i)= sudut(segment.c{i}(end,:),segment.c{par}(2,:),segment.c{i}(end,:));
%         end
%     end  
% end


% Compute the angle in bifurcation relative to parent segment
% function angle = sudut(P0,P1,P2) % P=[x,y,z], (mid,par,end)
% n1 = (P2 - P0) / norm(P2 - P0);  % Normalized vectors
% n2 = (P1 - P0) / norm(P1 - P0);
% angle = pi-atan2(norm(cross(n1, n2)), dot(n1, n2));  % Stable
% m1 = (P0(2)-P1(2))/(P0(1)-P1(1));
% m2 = (P2(2)-P0(2))/(P2(1)-P0(1));
% end


end