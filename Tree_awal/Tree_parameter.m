function prm = Tree_parameter(alpha)
%% Artery Parameters
% blood and flow parameters (all unit in cgs)
prm.Rho = 1;
prm.mu = 1;
prm.Pin = 1e6;           % capillary pressure
prm.Qin = 100;          % Flowrate in input node

% Geometry parameters
prm.Ro = 2;      % Initial radius 
prm.LtoR = alpha(3);    % Ratio between length and radius
prm.Mexp = alpha(9);    % Bifurcation exponent
prm.Rmin = 0.85*prm.Ro;  % Minimum radius
prm.N = 2;              % number of bifurcation
prm.a_init = [alpha(7),0];       % [x,y] initial point coordinate(root terminal)
prm.a_theta = alpha(5);         % initial theta respect to x-axis
gamma = alpha(1);              % Radius ratio between both childs vessel (degree of symmetry)
[prm.phi,prm.ratio]=phiratio(prm, gamma);

%% Vein Parameter
% Geometry parameters
prm.Rvo = alpha(4)*prm.Ro;       % Initial radius 
prm.Rvmin = 0.9*alpha(4)*prm.Rmin;   % Minimum radius
prm.v_init=[alpha(8),160];       % [x,y] initial point coordinate(root terminal)
prm.v_theta = alpha(6);         % initial theta respect to x-axis
gamma = alpha(2);                  % Radius ratio between both childs vessel (degree of symmetry)
[prm.vphi,prm.vratio]=phiratio(prm, gamma);
for i = 10:13
    gamma = alpha(i);
    [phi,ratio] = phiratio(prm, gamma);
    prm.phi = [prm.phi; phi];
    prm.ratio = [prm.ratio; ratio];
end
for i = 14:20
    gamma = alpha(i);
    [phi,ratio] = phiratio(prm, gamma);
    prm.vphi = [prm.vphi; phi];
    prm.vratio = [prm.vratio; ratio];
end
% Ratio between parent and 2 childs vessel
    function [phi,ratio] = phiratio(prm, gamma)
        delta = (1+gamma^(prm.Mexp/2))^(-1/prm.Mexp); 
        beta = delta*sqrt(gamma);
        phi(1) = acos((1+delta^4-beta^4)/(2*delta^2));    % angle between first and parent
        phi(2) = acos((1-delta^4+beta^4)/(2*beta^2));     % angle between second and parent
        ratio = [delta, beta];   % Radius ratio between 2 childs to the parent vessel
    end
end