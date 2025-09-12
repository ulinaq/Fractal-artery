function prm = Frog_parameter()
%% Artery Parameters
% blood and flow parameters (all unit in mmgs)
prm.Rho = 1e-6;       % kg/mm3
prm.mu = 3e-6;        % kg/mm.s
prm.perm = 3e-6;      % mm2
prm.Pin = 3.99967105; % kg/mm.s2
prm.Pout = 1;         % kPa
prm.pixel = 6e-2;     % milimeter
prm.alpha = 5e-4;     % Perfusion parameter
prm.beta = 2;         % relative permeability of vein compartment to artery compartment
prm.gamma = 5e-5;     % kg/mm4.s (resistance parameter around terminal nodes, kappa in the manuscript)
prm.E = 1e3;          % kPa (vein)/artery Ea = 1e3
prm.Poison = 0.5;     % Poisson's ratio
prm.ha = 0.22;        % arterial wall thickness coeff.
prm.hv = 0.1;         % vein wall thickness coeff.
prm.Por = 0.1;        % 
prm.mmHg = 7.50061683;
end