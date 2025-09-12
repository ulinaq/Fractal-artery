function prm = GAI_parameter(alpha)
%% Artery Parameters
% blood and flow parameters (all unit in cgs)
prm.Rho = 1;
prm.mu = 1;
prm.Pin = 1e6;           % capillary pressure
prm.Qin = 100;          % Flowrate in input node

% Geometry parameters
prm.Ro = 2;      % Initial radius 
prm.LtoR = 10;    % Ratio between length and radius
prm.Mexp = 2.8;    % Bifurcation exponent
prm.Rmin = 0.95*prm.Ro;  % Minimum radius
prm.N = 2;              % number of bifurcation
gamma = alpha(1);              % Radius ratio between both childs vessel (degree of symmetry)
prm.a_init = [70,0];       % [x,y] initial point coordinate(root terminal)
prm.a_theta = pi/2;         % initial theta respect to x-axis

% Ratio between parent and 2 childs vessel
delta = (1+gamma^(prm.Mexp/2))^(-1/prm.Mexp); 
beta = delta*sqrt(gamma);
prm.phi(1) = acos((1+delta^4-beta^4)/(2*delta^2));    % angle between first and parent
prm.phi(2) = acos((1-delta^4+beta^4)/(2*beta^2));     % angle between second and parent
prm.ratio = [delta, beta];   % Radius ratio between 2 childs to the parent vessel

%% Vein Parameters
% Geometry parameters
prm.Rvo = 1.2*prm.Ro;       % Initial radius 
prm.Rvmin = 1.2*prm.Rmin;   % Minimum radius
gamma = alpha(2);                  % Radius ratio between both childs vessel (degree of symmetry)
prm.v_init=[-70,160];       % [x,y] initial point coordinate(root terminal)
prm.v_theta = -pi/2;         % initial theta respect to x-axis

% Ratio between parent and 2 childs vessel
delta = (1+gamma^(prm.Mexp/2))^(-1/prm.Mexp); 
beta = delta*sqrt(gamma);
prm.vphi(1) = acos((1+delta^4-beta^4)/(2*delta^2));    % angle between first and parent
prm.vphi(2) = acos((1-delta^4+beta^4)/(2*beta^2));     % angle between second and parent
prm.vratio = [delta, beta];   % Radius ratio between 2 childs to the parent vessel
end