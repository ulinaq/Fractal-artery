figure()
subplot(1,2,1)
bar(Art_bif.no);
title('Artery');
xlabel('number of branches in bifurcation');
ylabel('frequency');

% figure()
subplot(1,2,2)
bar(Vein_bif.no);
title('Vein');
xlabel('number of branches in bifurcation');
ylabel('frequency');

figure()
histogram(l2r,'Normalization','pdf');
% figure()
hold on;
histogram(l2rv,'Normalization','pdf');
title('Length-radii ratio');
xlabel('Ratio');
ylabel('pdf');

figure()
histogram(at,'Normalization','probability');
hold on;
histogram(av,'Normalization','probability');
title('Bifurcation exponent');
xlabel('bifurcation exp.');
ylabel('probability');

amin= mean(ar2)-std(ar2);
amax= mean(ar2)+std(ar2);
vmin= mean(vr2)-std(vr2);
vmax= mean(vr2)+std(vr2);
figure()
histogram(ar2(ar2>amin & ar2<amax),'Normalization','pdf');
hold on;
histogram(vr2(vr2>vmin & vr2<vmax),'Normalization','pdf');
title('Bifurcation exponent for 2 branches');
xlabel('bifurcation exp.');
ylabel('probability');

figure()
histogram(ratio,'Normalization','probability');
% figure()
hold on;
histogram(vratio,'Normalization','probability');
title('Child-radii ratio');
xlabel('Ratio');
ylabel('probability');

figure()
histogram(Lratio,'Normalization','probability');
% figure()
hold on;
histogram(vLratio,'Normalization','probability');
title('Child-length ratio');
xlabel('Ratio');
ylabel('probability');

figure()
subplot(2,2,1)
histogram(ar2,'Normalization','pdf');
title('Artery (2 branches) 1^{st} scheme');
xlabel('bifurcation exp.');
ylabel('frequency');

% figure()
subplot(2,2,2)
histogram(ar3);
title('Artery (3 branches) 1^{st} scheme');
xlabel('bifurcation exp.');
ylabel('frequency');

subplot(2,2,3)
histogram(a2);
title('Artery (2 branches) 2^{nd} scheme');
xlabel('bifurcation exp.');
ylabel('frequency');

% figure()
subplot(2,2,4)
histogram(a3);
title('Artery (3 branches) 2^{nd} scheme');
xlabel('bifurcation exp.');
ylabel('frequency');

figure()
subplot(2,4,1)
histogram(vr2);
title('2 branches (1^{st})');
xlabel('bifurcation exp.');
ylabel('frequency');

% figure()
subplot(2,4,2)
histogram(vr3);
title('3 branches (1^{st})');
xlabel('bifurcation exp.');
ylabel('frequency');

subplot(2,4,3)
histogram(vr4);
title('4 branches (1^{st})');
xlabel('bifurcation exp.');
ylabel('frequency');

% figure()
subplot(2,4,4)
histogram(vr569);
title('>4 branches (1^{st})');
xlabel('bifurcation exp.');
ylabel('frequency');

subplot(2,4,5)
histogram(v2);
title('2 branches (2^{nd})');
xlabel('bifurcation exp.');
ylabel('frequency');

% figure()
subplot(2,4,6)
histogram(v3);
title('3 branches (2^{nd})');
xlabel('bifurcation exp.');
ylabel('frequency');

subplot(2,4,7)
histogram(v4);
title('4 branches (2^{nd})');
xlabel('bifurcation exp.');
ylabel('frequency');

% figure()
subplot(2,4,8)
histogram(v569);
title('>4 branches (2^{nd})');
xlabel('bifurcation exp.');
ylabel('frequency');