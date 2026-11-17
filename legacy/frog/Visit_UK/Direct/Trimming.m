Bflow = Bflow200;
% Bflow = Bflo{9};
[~,tem] = find(Bflow(1,:)==0);
Bflow(:,unique(tem))=[];
% Bflow(1,:) = Bflow(1,:)*0.55;
b =zeros(size(Bflow));
for i=1:length(Bflow)
b(1:5,i)=Bflow(1:5,i)/Bflow(1,i);
b(6:8,i)=Bflow(6:8,i)/Bflow(6,i);
end
[~,tem] = find(b>100 | b<0);
b(:,unique(tem))=[];
% Flmean = mean(b,2);
% Flmean(:,2) = std(b,0,2)
bb = b(:,randperm(length(b),200));