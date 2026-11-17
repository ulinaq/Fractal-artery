as=zeros(length(NN),4);
for i=1:length(NN)
    as(i,:)=varRV2{i,2};
end
as(20,1)=sum(x([Co+segment.num(segment.isroot)]));
as(20,2)=(prm.Pin-prm.Pout)/sum(x([Co+segment.num(segment.isroot)]));

x=95:-5:0;


figure(1)
clf;
plot(x,Rootva{1}(:,1),x,Rootva{2}(:,1),x,Rootva{3}(:,1),x,Rootva{4}(:,1),'LineWidth',2); 
title('Total flow');  
xlabel('vessel closure [%]')
ylabel('Q [mm^3/s]')
legend('artery-1','artery-2','vein-1','vein-2')
subplot(2,2,2);
plot(x,Rootva{1}(:,2),x,Rootva{2}(:,2),x,Rootva{3}(:,2),x,Rootva{4}(:,2),'LineWidth',2); 
title('Total resistance');  
xlabel('vessel closure [%]')
ylabel('R [kg.mm^{-4}s^{-1}]')
legend('artery-1','artery-2','vein-1','vein-2')
subplot(2,2,3);
semilogy(x,Rootva{1}(:,3),x,Rootva{2}(:,3),x,Rootva{3}(:,3),x,Rootva{4}(:,3),'LineWidth',2); 
title('Average arterial vessel resistance');  
xlabel('vessel closure [%]')
ylabel('R [kg.mm^{-4}s^{-1}]')
legend('artery-1','artery-2','vein-1','vein-2')
subplot(2,2,4);
semilogy(x,Rootva{1}(:,4),x,Rootva{2}(:,4),x,Rootva{3}(:,4),x,Rootva{4}(:,4),'LineWidth',2); 
title('Average vein vessel resistance');  
xlabel('vessel closure [%]')
ylabel('R [kg.mm^{-4}s^{-1}]')
legend('artery-1','artery-2','vein-1','vein-2')

as=latex(vpa(sym([mean(Pan(Grid.B)) std(Pan(Grid.B)) mean(Pvn(Grid.B)) std(Pvn(Grid.B))]),4))

latex_table = latex(vpa(sym(Rootva{1}(1,:)),4))
