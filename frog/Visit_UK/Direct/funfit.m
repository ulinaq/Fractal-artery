function [Pfit,fit, resnorm] = funfit(Time, aif, roi, fit_model,varargin)
% Input argument

% if nargin>4
%     PF = varargin{1};
%     PV = varargin{2};
% else
    PF = 25;
    PV = 120;
% end

Hct = 0.45; % #Haematocrit factor
bL = 9; % # Baseline, default=9
r = 3.6; % # Relaxivity in Hz/mM, used in function Concentration_SPGRESS
s = [-3,3,0.02]; % # Delay values: minimum, maximum and step size, used in function add_delay

delay_condition = true;%false;% % # # When true adds arterial delay to the linear model 
roi0 = mean(roi(1:bL)); %
aif0 =  mean(aif(1:bL)); %
roi_new = roi-roi0/(1-Hct); %  # Signal for kidney after baseline correction
aif_new = (aif-aif0)/(1-Hct); % # AIF after baseline and haematocrit correction 'Algorithm','levenberg-marquardt',
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt',...
        'MaxFunctionEvaluations',200000,'FunctionTolerance',1e-12,'StepTolerance',1e-12,...
        'MaxIterations',200000);
%,'Algorithm','levenberg-marquardt'
%## Fitting concentration with chosen model ##
switch fit_model
      case 1    % 2CM uptake model
          P = [PF/100, PV/(6000), 12/132.]; %[VP, FP, FE/(FP+FE)]
          if delay_condition == true
              [res, fit, resnorm, Delay] = add_delay(P,Time, aif_new,roi_new,s,fit_model,options);
              Pfit.Delay = Delay;
          else
              fun = @(x,aif_new)model_uptake(x,Time, aif_new);
              [res, ~, resnorm] = lsqcurvefit(fun, P,aif_new,roi_new,[],[],options);%    options);
              fit = model_uptake(res,Time, aif_new);
          end
          Pfit.BF = 6000*res(2)*(1 -Hct); %	#Blood Flow
          Pfit.BV = 100*res(1); % #Blood Volume
          Pfit.BMTT = 1*res(1)*(1-res(3))/res(2); % #Blood MTT
          Pfit.PS = 6000*res(2)*res(3)/(1-res(3));  %Permeability-surface area product
          Pfit.Ktrans = 6000*res(2)*res(3);
          Pfit.EF = 100*res(3); % #Extraction Fraction
      case 2    % 2CM exchange model
          P = [20/100, 100/(6000*(1-Hct)), 0.5/3, 0.3]; % #[VP+VE, FP, VE/(VP+VE), PS/FP]
          if delay_condition == true
              [res, fit, resnorm, Delay] = add_delay(P,Time, aif_new,roi_new,s,fit_model,options);
              Pfit.Delay = Delay;
          else
              fun = @(x,aif_new)model_2CXM(x,Time, aif_new);
              [res,~, resnorm] = lsqcurvefit(fun, P,aif_new,roi_new,[],[],options);
              fit = model_2CXM(res,Time,aif_new);        
          end
          Pfit.BF = 6000*res(2)*(1 -Hct); %	#Blood Flow
          Pfit.BV = 100*res(1)*(1-res(3)); % #Blood Volume
          Pfit.BMTT = 1*res(1)*(1-res(4))*res(1)*(1-res(3))/res(2); % #Blood MTT
          Pfit.PS = 6000*res(2)*res(4)/(1-res(4));  %Permeability-surface area product
          Pfit.IV = 100*P(1)*P(3); % #Interstitial Volume
          Pfit.IMTT = 1*res(1)*res(3)*(1-res(4))/(res(2)*res(4)); % #Tubular MTT
          Pfit.Ktrans = 6000*res(2)*res(4);
          Pfit.Kep = 6000*res(2)*res(4)/(res(1)*res(3));
          Pfit.EF = 100*res(4); % #Extraction Fraction
          Pfit.EV =  100*res(1);%	#Extracellular Volume
          Pfit.EMTT = 1*res(1)/res(2);  %Extracellular MTT
      case 3    % Maximum Slope model
          [fit,flow] = MaximumSlope(Time, roi_new, aif_new);
          Pfit.BF = flow;%*(1-Hct);
          resnorm = [];
      case 4    % Compartment model
          P = [PF/100, PV/(6000*(1-Hct))]; %;[V, F]
          if delay_condition == true
              [res, fit, resnorm, Delay] = add_delay(P,Time, aif_new,roi_new,s,fit_model,options); 
              Pfit.Delay = Delay;
          else
              fun = @(x,aif_new)model_CM(x,Time, aif_new);
              [res,~, resnorm] = lsqcurvefit(fun, P,aif_new,roi_new,[],[],options);
              fit = model_CM(res,Time,aif_new);        
          end
          Pfit.BF = 6000*res(2)*(1 -Hct); %	#Blood Flow
          Pfit.EV =  100*res(1);%	#Extracellular Volume
          Pfit.EMTT = 1*res(1)/res(2);  %Extracellular MTT
end

%% Functions
% #ShiftAif(ca, time, Delay) shifts the arterial concentration (ca) curve by a delay time using
% # the formula ca(shifted) = ca(Time - Delay).

function ShAif = ShiftAif(ca, time, Delay)
    ShAif = interp1(time, ca, time-Delay);
    nneg = sum(time <= Delay);
    ShAif(isnan(ShAif)) = min(ShAif);
    if nneg > 0
        ShAif(1:nneg-1) = 0;
    end
end


% # INT_TRAP(t, c) integrates vector c with respect to t using trapezoidal rule. 

function trapz_int_final = INT_TRAP(t,c)
    n = length(t);
    dt = t(2:n)-t(1:n-1);
    trapz_int = dt.*(c(1:n-1)+c(2:n))/2;
    trapz_int_2 = [0;trapz_int]; 
    trapz_int_final = cumsum(trapz_int_2);
end 

% # EXPCONV(T, time, ca) evaluates convolution of ca with exp(-time/T)/T. More details can be found 
% #in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006

function [f,DY] = EXPCONV(T, time, ca, Der)
if nargin == 3
    n = length(time);
    f = zeros(n,1);
    x = (time(2:n) - time(1:n-1));
    da = (ca(2:n) - ca(1:n-1));
    Z = T*x;
    E = exp(-Z);
    E0 = 1-E;
    E1 = Z-E0;
    add = (ca(1:n-1).*E0 + da.*E1./Z)/T;
    for i = 1 : n-1 %in range(0,n-1):
        f(i+1) = E(i)*f(i) + add(i);
    end
else
    E2 = Z.^2-2*E1;
    DIl = -x*add + (ca(1:n-1).*E1 + da*E2./Z )/(T^2);
    DIl = -E.*x.*f + DIl;

	DD = zeros(n,n);

	for i=1:n-1
        DY(i+1) = E(i)*DY(i) + DIl(i);
    end
end
    if nargout > 1
        DY = DD;
    end
end
 
% # model_MaxSlope(P, time, ca) evaluates concentration-time curve for a Maximum slope model 
function [Fit,flow] = MaximumSlope(time, curve, AIF)
	[cMaxDer,i] = max(deriv(time,curve));
	Fit = curve(i) + cMaxDer * (time - time(i));
	i = find(Fit > max(curve));
    if ~isempty(i)
        Fit(i) = max(curve);
    end
	i = find(Fit < min(curve));
    if ~isempty(i)
        Fit(i) = min(curve);
    end
    flow = 6000*cMaxDer/max(AIF);
end

function d = deriv(x,y)
    x12 = x - circshift(x,-1)   ;%x1 - x2
    x01 = circshift(x,1) - x    ;%x0 - x1
    x02 = circshift(x,1) - circshift(x,-1) ;%x0 - x2
% middle    
    d = circshift(y,1).*(x12./(x01.*x02)) + y.*(1./x12 - 1./x01) - ...
        circshift(y,-1).*(x01./(x02.*x12));
% ; Formulae for the first and last points:
    d(1) = y(1)* (x01(2)+x02(2))/(x01(2)*x02(2)) - y(2) * x02(2)/(x01(2)*x12(2)) + ...
        y(3) * x01(2)/(x02(2)*x12(2));
    d(end) = -y(end-2) * x12(end-1)/(x01(end-1)*x02(end-1)) + ...
        y(end-1) * x02(end-1)/(x01(end-1)*x12(end-1)) - ...
        y(end) * (x02(end-1)+x12(end-1)) / (x02(end-1)*x12(end-1));
end

% One-Compartment Model
function C = model_CM(P, time, ca)
    % P = (V,F)
    K = P(2)/P(1);
	C = P(2)*EXPCONV(K, time, ca);
end

% # model_2CXM(P, time, ca) evaluates concentration-time curve for a two-compartment exchange model (2CXM) 
% #following the formula in Eq. 7 in Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006. 
% #Where P = [VP+VE, FP, VE/(VP+VE), PS/FP], tissue parameters, ca = arterial concentration.

function conc = model_2CXM(P, time, ca) % Exchange model
    A = P(3)-P(3)*P(4)+P(4);
	B = P(3)*(1-P(3))*P(4)*(1-P(4));

    ts = sqrt(1-4*B/A^2);
	ta = 0.5*A/P(4);

    tp = ta*(1+ts);
	tm = ta*(1-ts);

	Kp = P(2)/(P(1)*tm);
	Km = P(2)/(P(1)*tp);
	Am = (1-tm)/(tp-tm);

	conc = P(2)*(1-Am)*EXPCONV(Kp,time,ca) + P(2)*Am*EXPCONV(Km,time,ca);
end

% #model_uptake(P, time, ca) evaluates concentration-time curve for a two-compartment uptake model
% #following the formula ct(t) = (1-E) FP exp(-t/TP) (convolution) ca(t) + E FP (convolution) ca(t)
% #where P = [VP, FP, VE/(VP+VE), PS/FP], E = PS/FP, TP = VP/FP. 
% [VP, FP, FE/(FP+FE)]
function conc = model_uptake(P, time, ca)
    Fp = P(2);
%     PS = P(4)*Fp;
    PS = P(3)*Fp;
    VE = P(3)*P(1);
    VP = P(1)-VE;
%    TP = Fp/VP;
    TP = P(2)/(P(1)*(1-P(3)));
    E = PS/Fp;
%     conc = Fp*(1-E)*TP*EXPCONV(TP, time, ca) + Fp*E*INT_TRAP( time, ca);
    conc = Fp*(1-P(3))*EXPCONV(TP, time, ca) + Fp*P(3)*INT_TRAP( time, ca);
end

% #delay_error(roi,Delay,dt,nT,fit_shft) calculates the error of
% #the concentration vs time curve shifted with a delay time against the original curve.
% #Delay = delay time
% #roi = tracer concentration in the tissue
% #dt = time interval
% #nT = number of time points
% #fit_shft = shifted curve

function Error =  delay_error(roi,Delay,dt,nT,fit_shft)
    nd = abs((round(Delay/dt)));
    Error =  100*sum((roi(1:nT-nd)-fit_shft(1:nT-nd)).^2)/(sum(roi(1:nT-nd).^2));
end

% #add_delay(P,time, ca,roi_new,shift,nid,fit_model) determines the delay value by fitting the 
% #concentration vs time curve for a range 
% #of delay values and seleting the one for which the least-square error is minimum.  It returns
% #the fitted tissue parameters and tissue curve for the chosen delay value. 
% #Arguments
% #roi = tracer concentration in the tissue
% #shift = array of three values- minimum value of delay, maxmum value of delay and step size.
% #nid = patient id. 
% #fit_model = Model used to fit the C-t curve, either 1 or 2
    
function [res, fit, Err, Delay] = add_delay(P,time, ca,roi,shift,fit_model,options)
    n = 1+(shift(2)-shift(1))/shift(3);
    Delay = shift(1) + shift(3)*(0:n-1);
    Error = Delay*0;
    dt = time(2)-time(1);
    nT = length(time);
%     options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt',...
%         'MaxFunctionEvaluations',20000,'FunctionTolerance',1e-8,'StepTolerance',1e-8);

    for i = 1:round(n)
        if Delay(i) > 0
            aif_del = ShiftAif(ca,time,Delay(i));
        else
            aif_del = ca; 
        end
        if Delay(i) < 0 
            roi_del = ShiftAif(roi,time,-Delay(i)); 
        else
            roi_del = roi;    
        end
        switch fit_model
            case 1
                fun = @(x,aif_del)model_uptake(x,time, aif_del);
                [res, resnorm] = lsqcurvefit(fun, P,aif_del,roi_del,[],[],options);
                fit = model_uptake(res,time, aif_del);
            case 2
                fun = @(x,aif_del)model_2CXM(x,time, aif_del);
                [res, resnorm] = lsqcurvefit(fun, P,aif_del,roi_del,[],[],options);
                fit = model_2CXM(res,time,aif_del);
            case 4
                fun = @(x,aif_new)model_CM(x,time, aif_del);
                [res, resnorm] = lsqcurvefit(fun, P,aif_del,roi_del,[],[],options);
                fit = model_CM(res,time,aif_del);        
        end
        if Delay(i) < 0 
            fit_shft = ShiftAif(fit,time,Delay(i)); 
        else
            fit_shft = fit;
        end
        Error(i) =  delay_error(roi,Delay(i),dt,nT,fit_shft);
    end
           
    [Err,imin] = min(Error);
    Delay = Delay(imin);
    msg = ['Delay =', num2str(Delay)];
    disp(msg);
    if Delay > 0 
        aif_del = ShiftAif(ca,time,Delay); 
    else
        aif_del = ca;
    end
    if Delay < 0 
        roi_del = ShiftAif(roi,time,-Delay); 
    else
        roi_del = roi;
    end
    switch fit_model
            case 1
                fun = @(x,aif_del)model_uptake(x,time, aif_del);
                [res,~, resnorm] = lsqcurvefit(fun, P,aif_del,roi_del,[],[],options);
                fit = model_uptake(res,time, aif_del);
            case 2
                fun = @(x,aif_del)model_2CXM(x,time, aif_del);
                [res,~, resnorm] = lsqcurvefit(fun, P,aif_del,roi_del,[],[],options);
                fit = model_2CXM(res,time,aif_del);
            case 4
                fun = @(x,aif_new)model_CM(x,time, aif_del);
                [res,~, resnorm] = lsqcurvefit(fun, P,aif_del,roi_del,[],[],options);
                fit = model_CM(res,time,aif_del);        
    end            
    if Delay < 0
        fit = interp1(time, fit,time-Delay);
    end
end                
end