clear all;
clc;

n=5;    LtoR=20;    Mexp=2.7;   gamma=1;  
Ro=4;   xb=[0,0];   yb=[0,LtoR*Ro];     % main vessel (first trunk of tree)

delta=(1+gamma^(Mexp/2))^(-1/Mexp); 
beta=delta*sqrt(gamma);
phi(1)=acos((1+delta^4-beta^4)/(2*delta^2));
phi(2)=acos((1-delta^4+beta^4)/(2*beta^2));
rM=[delta, beta];


h=figure;
 axes('Parent',h);
 daspect([1 1 1])

mN=length(phi);   
psi=zeros(n,mN^n);  % angles in each bifurcation
ralt=ones(n,mN^n);  % ratio between child and parent vessel in bifurcation
  %   psi and ralt have Kantor array`s structur !!!
   
% The adress to the system to ignore warnings as divide by zero 
  warning off MATLAB:divideByZero
   
 % ksi - angle of trunk with vertical axes
 % it`s calculated on the basis of coordinates xb and yb of trunk
 %  by using the line-equation:
 %  y=k*x+d,   here
 %      k=(yb(2)-yb(1))/(xb(2)-xb(1));
 %      d=(yb(1)*xb(2)-xb(1)*yb(2))/(xb(2)-xb(1));
         a=sqrt((xb(1)-xb(2))^2+(yb(1)-yb(2))^2);   % scale factor
         b=1;
         c=sqrt((1-xb(2))^2+(yb(2))^2);
         alpha=acos((a^2+b^2-c^2)/(2*a*b));
         k=(yb(2)-yb(1))/(xb(2)-xb(1));                       % define the tangent
         d=(yb(1)*xb(2)-xb(1)*yb(2))/(xb(2)-xb(1));           % define the free member
             if yb(2)>=yb(1)
                   ksi=alpha;
               else
                   ksi=-alpha; 
             end
             
 
    for i=1:1:n
        z=1;
       for j=1:1:mN^(i-1)
           kpi=randperm(mN);                           % pick random number between 1 and 2
           for k=1:1:mN                           
               for m=1:1:mN^(n-i)
                   psi(i,z)=(-1)^k*phi(kpi(k));        % define psi on the base of phi, one angle to the right child vessel and the next one for left child vessel
                   ralt(i,z)=rM(kpi(k));               % define ralt on the base of rM  
                   z=z+1;
               end
           end
       end
   end
  
   theta=zeros(n,mN^n);  % vector of angles between each branch of fractal and vertical axes
   rD=ones(n,mN^n);      % lengths of branches
   
   for i=1:1:mN^n
       for j=1:1:n
           for k=1:1:j
           theta(j,i)=theta(j,i)+psi(k,i);        % define theta on the base of psi
           rD(j,i)=rD(j,i)*ralt(k,i);             % define rD on the base of rD
       end
       end
   end
   
   theta=theta+ksi;
    
         
% ----Matrix for coordinates-------         
   A=ones(n+1,mN^n,2);
% initial coordinates   
   A(1,:,1)=ones(1,mN^n)*xb(2);   % x-coordinate
   A(1,:,2)=ones(1,mN^n)*yb(2);   % y-coordinate
      for j=1:1:mN^n
       for i=1:1:n
           
           % define following coordinates
           A(i+1,j,1)=A(i,j,1)+a*rD(i,j)*cos(theta(i,j));
           A(i+1,j,2)=A(i,j,2)+a*rD(i,j)*sin(theta(i,j));
       end
   end
     
 % By visualisation the method of inverse trace is used    
    tau=1;
      for i=1:mN:mN^n
          z=1;
           for k=1:1:mN-1
                for j=z:1:n-1
                 line([A(j,i,1),A(j+1,i,1)],[A(j,i,2),A(j+1,i,2)],'Color',[0.8,0.5,0.5],'LineWidth',Ro*rD(j,i));
                end
              z=z+1;
          end       
     end
 
 % -------------------------- end branches ----------------------------    
      for i=1:1:mN^n
        line([A(n,i,1),A(n+1,i,1)],[A(n,i,2),A(n+1,i,2)],'Color',[0.3,0.75,0.4],'LineWidth',Ro*rD(n,i));
      end   
 % -------------- Trunk -----------------         
    line([xb(1),xb(2)],[yb(1),yb(2)],'Color',[0.8,0.5,0.5],'LineWidth',Ro);
        
