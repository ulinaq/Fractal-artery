function prm = Frog_parameter()
%% Artery Parameters
% blood and flow parameters (all unit in cgs)
prm.Rho = 1;
prm.mu = 1;
prm.Pin = 1e5;           % capillary pressure
prm.Qin = 5e-1;          % Flowrate in input node
prm.pixel = 3.2e-1;         % micrometer

% Geometry parameters
prm.Ro = 2;      % Initial radius 
prm.LtoR = 10;    % Ratio between length and radius
prm.Mexp = 2.8;    % Bifurcation exponent
prm.Rmin = 0.85*prm.Ro;  % Minimum radius
prm.N = 2;              % number of bifurcation
prm.a_init = [35,0];       % [x,y] initial point coordinate(root terminal)
prm.a_theta = pi/2;         % initial theta respect to x-axis
gamma = 1.4;              % Radius ratio between both childs vessel (degree of symmetry)
[prm.phi,prm.ratio]=phiratio(prm, gamma);

%% Vein Parameter
% Geometry parameters
prm.Rvo = 1.2*prm.Ro;       % Initial radius 
prm.Rvmin = 1.2*prm.Rmin;   % Minimum radius
prm.v_init=[30,0];       % [x,y] initial point coordinate(root terminal)
prm.v_theta = pi/2;         % initial theta respect to x-axis
gamma = 1.2;                  % Radius ratio between both childs vessel (degree of symmetry)
[prm.vphi,prm.vratio]=phiratio(prm, gamma);
for i = 1:5
    gamma = rand(1);
    [phi,ratio] = phiratio(prm, gamma);
    prm.phi = [prm.phi; phi];
    prm.ratio = [prm.ratio; ratio];
end
for i = 1:5
    gamma = rand(1);
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