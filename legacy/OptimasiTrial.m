lb = [0.3,0.3,10,1,pi/2,-pi/2];
ub = [10/3,10/3,30,3,pi,0];

% choose option
optPSw1 = optimoptions('particleswarm', 'UseParallel', true, 'UseVectorized', false);
optPSwpar = optimoptions('particleswarm', 'UseParallel', true, 'UseVectorized', false);
optPSepar = optimoptions('patternsearch','UseParallel', true, 'UseCompletePoll', true, 'UseVectorized', false);
N=10;
for i=1:N
x0 = rand(1,6).*(ub-lb)+lb;
%     [xptsc{i},fval(i)]=patternsearch(@OptimTree,x0,[],[],[],[],lb,ub,optPSepar);
%     [xpsw{i},fvalsw(i)]=fmincon(@OptimTree,x0,[],[],[],[],lb,ub);
[xpsw{i},fvalsw(i)]=particleswarm(@OptimTree,6,lb,ub,optPSwpar);
end

xaRange=particleswarm(@OptimTreeRange,6,lb,ub,optPSwpar)
xaL2=particleswarm(@OptimTreeL2,6,lb,ub,optPSwpar)
xaPout=particleswarm(@OptimTree,6,lb,ub,optPSwpar)
xaSTD=particleswarm(@OptimTreeSTD,6,lb,ub,optPSwpar)
xaMean=particleswarm(@OptimTreeMean,6,lb,ub,optPSwpar)

xa=[OptimTree(xaRange) OptimTree(xaL2) OptimTree(xaPout) OptimTree(xaSTD) OptimTree(xaMean);...
    OptimTreeRange(xaRange) OptimTreeRange(xaL2) OptimTreeRange(xaPout) OptimTreeRange(xaSTD) OptimTreeRange(xaMean);...
    OptimTreeL2(xaRange) OptimTreeL2(xaL2) OptimTreeL2(xaPout) OptimTreeL2(xaSTD) OptimTreeL2(xaMean);...
    OptimTreeSTD(xaRange) OptimTreeSTD(xaL2) OptimTreeSTD(xaPout) OptimTreeSTD(xaSTD) OptimTreeSTD(xaMean);...
    OptimTreeMean(xaRange) OptimTreeMean(xaL2) OptimTreeMean(xaPout) OptimTreeMean(xaSTD) OptimTreeMean(xaMean)];



xaToFmax4=particleswarm(@OptimTreeToF,6,lb,ub,optPSwpar)
V=Tree_main(xaToFmax4)
xaToFstd=particleswarm(@OptimTreeToF,6,lb,ub,optPSwpar)
V=Tree_main(xaToFstd)
xaToFL2=particleswarm(@OptimTreeToF,6,lb,ub,optPSwpar)
V=Tree_main(xaToFL2)
xaToFRa=particleswarm(@OptimTreeToF,6,lb,ub,optPSwpar)
V=Tree_main(xaToFRa)
xaToFmaxP2=particleswarm(@OptimTreeToF,6,lb,ub,optPSwpar)
V=Tree_main(xaToFmaxP2)

% add alpha(7) and (8) for root terminal variables
lb = [0.3,0.3,10,1,0,-pi,-70,-70];
ub = [10/3,10/3,30,3,pi,0,-70,70];

xaToFmax=particleswarm(@OptimTreeToF,8,lb,ub,optPSwpar)
V=Tree_main(xaToFmax)