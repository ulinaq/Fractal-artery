function prm = Frog_parameter(comp)
%% Artery Parameters
% blood and flow parameters (all unit in mmgs)
prm.Comp = comp;
if comp~=1 && comp~=2
    msg = 'Error, input 1 or 2, to choose the number of compartments';
    error(msg);
end
prm.Rho = 1e-6;       % kg/mm3
prm.mu = 3e-6;        % kg/mm.s
prm.perm = 3e-7;      % mm2
prm.Pin = 10;         % kg/mm.s2
prm.Pout = 2.5;         % kPa
prm.pixel = 6e-2;     % milimeter
prm.alpha = 5e-5;     % 
prm.beta = 1;         % relative permeability of vein compartment to artery compartment
prm.gamma = 1e-8;     % mm3 (resistance parameter around terminal nodes)
prm.E = 1000;         % kPa
prm.Poison = 0.5;     % Poisson's ratio
prm.ha = 0.1;           % arterial wall thickness coeff.
prm.hv = 0.05;          % vein wall thickness coeff.
prm.Por = 0.1;
prm.type = 3;
prm.res = 2;         % Resistance coeffiecient in the direct connection 


% Geometry parameters
prm.Ro = 2;      % Initial radius 
prm.LtoR = 10;    % Ratio between length and radius
prm.Mexp = 2.8;    % Bifurcation exponent
prm.Rmin = 0.95*prm.Ro;  % Minimum radius
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