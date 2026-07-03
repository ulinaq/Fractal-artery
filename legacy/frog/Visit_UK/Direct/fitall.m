%% Functions
% # INT_TRAP(t, c) integrates vector c with respect to t using trapezoidal rule. 

function trapz_int_final = INT_TRAP(t,c)
    n = length(t);
    dt = t(2:n)-t(1:n-1);
    trapz_int = dt*(c(1:n-1)+c(2:n))/2;
    trapz_int_2 = [trapz_int,0]; 
    trapz_int_final = np.cumsum(trapz_int_2);
end 

% # EXPCONV(T, time, ca) evaluates convolution of ca with exp(-time/T)/T. More details can be found 
% #in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006

function f = EXPCONV(T, time, ca)
    n = length(time);
    f = zeros(n,1);
    x = (time(2:n) - time(1:n-1))/T;
    da = (ca(2:n) - ca(1:n-1))/x;
    E = exp(-x);
    E0 = 1-E;
    E1 = x-E0;
    add = ca(1:n-1)*E0 + da*E1;
    for i = 1 : n-1 %in range(0,n-1):
        f(i+1) = E(i)*f(i) + add(i);
    end
end

% # model_2CFM(P, time, ca) evaluates concentration-time curve for a two-compartment filtration model (2CFM) 
% #following the formula in Eq. 7 in Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006. 
% #Where P = [VP+VE, FP, VE/(VP+VE), PS/FP], tissue parameters, ca = arterial concentration.

function conc = model_2CFM(P, time, ca)
    Fp = P(2);
    PS = P(4)*Fp;
    VE = P(3)*P(1);
    VP = P(1)-VE;
    TP = VP/Fp;
    TE = VE/PS;
    TT = (VP+VE)/Fp;
    Tpos = TE;
    Tneg = TP;
    if Tpos == Tneg
        conc=Fp*Tpos*EXPCONV(Tpos, time, ca);
    else
        Epos = (TT-Tneg)/(Tpos-Tneg);
        Eneg = 1-Epos;
        conc = Fp*Epos*Tpos*EXPCONV(Tpos, time, ca) + Fp*Eneg*Tneg*EXPCONV(Tneg, time, ca);
    end
end

% #fun_2CFM(P, time, ct, ca) calculates the difference between concentration evaluated by model_2CFM
% #and measurec concentration (ct).

function conc = fun_2CFM(P, time, ct, ca)
     conc = model_2CFM(P, time, ca) - ct;
end
% #model_uptake(P, time, ca) evaluates concentration-time curve for a two-compartment uptake filtration model
% #following the formula ct(t) = (1-E) FP exp(-t/TP) (convolution) ca(t) + E FP (convolution) ca(t)
% #where P = [VP+VE, FP, VE/(VP+VE), PS/FP], E = PS/FP, TP = VP/FP. 

function conc = model_uptake(P, time, ca)
    Fp = P(2);
    PS = P(4)*Fp;
    VE = P(3)*P(1);
    VP = P(1)-VE;
    TP = VP/Fp;
    E = PS/Fp;
    conc = Fp*(1-E)*TP*EXPCONV(TP, time, ca) + Fp*E*INT_TRAP( time, ca);
end
% #fun_uptake(P, t,ct, ca) calculates the difference between concentration evaluated by model_uptake
% #and measurec concentration (ct).
    
function conc = fun_uptake(P, time,ct, ca)
    conc = model_uptake(P, time, ca) - ct;
end

% #Concentration_SPGRESS(S, S0, T10, FA, TR, r) calculates the tracer concentration ct from a 
% #signal S and a precontrast signal S0 measured in the steady state of a T1-weighted 
% #spoiled gradient-echo sequence
% #ARGUMENTS
% #S = Signal S(ct) at concentration ct
% #S0 = Precontrast signal S(ct=0)
% #FA = Flip angle in degrees
% #TR = Repetition time in msec (=time between two pulses)
% #Precontrast T10 in msec
% #r= Relaxivity in Hz/mM

function SPgress = Concentration_SPGRESS(S, S0, T10, FA, TR, r)
    E = exp(-TR/T10);
    ct = cos(FA*pi/180);
    Sn = (S/S0)*(1-E)/(1-ct*E);%	#normalized signal
    R1 = -log((1-Sn)/(1-ct*Sn))/TR;	%#relaxation rate in 1/msec
    SPgress = 1000*(R1 - 1/T10)/r;
end

% #ShiftAif(ca, time, Delay) shifts the arterial concentration (ca) curve by a delay time using
% # the formula ca(shifted) = ca(Time - Delay).
% 

function ShAif = ShiftAif(ca, time, Delay)
    ShAif = interp1(time-Delay, time, ca);
    nneg = sum(time <= Delay);
    if nneg > 0
        ShAif(1:nneg-1) = 0;
    end
end
% #delay_error(roi,Delay,dt,nT,fit_shft) calculates the error of
% #the concentration vs time curve shifted with a delay time against the original curve.
% #Delay = delay time
% #roi = tracer concentration in the tissue
% #dt = time interval
% #nT = number of time points
% #fit_shft = shifted curve

function Error =  delay_error(roi,Delay,dt,nT,fit_shft)
    nd = abs(int(round(Delay/dt)));
    Error =  100*sum((roi(1:nT-nd)-fit_shft(1:nT-nd))^2)/(sum(roi(1:nT-nd)^2));
end

% #add_delay(P,time, ca,roi_new,shift,nid,fit_model) determines the delay value by fitting the 
% #concentration vs time curve for a range 
% #of delay values and seleting the one for which the least-square error is minimum.  It returns
% #the fitted tissue parameters and tissue curve for the chosen delay value. 
% #Arguments
% #roi = tracer concentration in the tissue
% #shift = array of three values- minimum value of delay, maxmum value of delay and step size.
% #nid = patient id. 
% #fit_model = Model used to fit the C-t curve, either 'uptake' or '2CFM'
    
function [res, fit] = add_delay(P,time, ca,roi,shift,fit_model)
    n = 1+(shift(2)-shift(1))/shift(3);
    Delay = shift(1) + shift(3)*(0:n);
    Error = Delay*0;
    dt = time(2)-time(1);
    nT = length(time);
    options = optimset('MaxFunEvals',20000);
   
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
        if fit_model == 'uptake'
            fun = @(x)fun_uptake(x,time, roi_del,aif_del);
            res = fminsearch(fun, P,options);
            fit = model_uptake(res,time, aif_del);
        elseif fit_model == '2CFM'
            fun = @(x)fun_2CFM(x,time, roi_del,aif_del);
            res = fminsearch(fun, P,options);
            fit = model_2CFM(res,time,aif_del);
        end
        if Delay(i) < 0 
            fit_shft = ShiftAif(fit,time,Delay(i)); 
        else
            fit_shft = fit;
        end
        Error(i) =  delay_error(roi,Delay(i),dt,nT,fit_shft);
    end
           
    imin = Error==min(Error);
    Delay = Delay[imin];
    msg = ['Delay' Delay];
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
    if fit_model == 'uptake'
        fun = @(x)fun_uptake(x,time, roi_del,aif_del);
        res = fminsearch(fun, P,options);
        fit = model_uptake(res,time, aif_del);
    elseif fit_model == '2CFM'
        fun = @(x)fun_2CFM(x,time, roi_del,aif_del);
        res = fminsearch(fun, P,options);
        fit = model_2CFM(res,time,aif_del);
    end            
    if Delay < 0
        fit = np.interp(time-Delay, time, fit);
    end
end