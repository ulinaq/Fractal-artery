% bifurcation exponent
load('vessel-struct.mat');
locA=cell2mat(tree_a.segment.nodeconn);
locV=cell2mat(tree_v.segment.nodeconn);
Art_bif=bif(locA);
Vein_bif=bif(locV);

function Nbif = bif(locA)
n2=[]; n2i=[];
n3=[]; n3i=[];
n4=[]; n4i=[];
n5=[]; n5i=[];
n6=[]; n6i=[];
nElsi=[]; nEls={};
for i=1:length(locA)
    ac=find(locA(:,2)==locA(i,1));
    if ~isempty(ac)
        if nnz(ac)==2
        n2(end+1,:)=find(locA(:,2)==locA(i,1));
        n2i(end+1)=i;
        else if nnz(ac)==3
            n3(end+1,:)=find(locA(:,2)==locA(i,1));
            n3i(end+1)=i;
            else if nnz(ac)==4
                n4(end+1,:)=find(locA(:,2)==locA(i,1));
                n4i(end+1)=i;
                else if nnz(ac)==5
                        n5(end+1,:)=find(locA(:,2)==locA(i,1));
                        n5i(end+1)=i;
                    else if nnz(ac)==6
                        n6(end+1,:)=find(locA(:,2)==locA(i,1));
                        n6i(end+1)=i;
                        else
                            nEls{end+1}=find(locA(:,2)==locA(i,1));
                            nElsi(end+1)=i;
                    end
            end
        end
    end
end
end
if nnz(n2)~=0
    Nbif.n2=n2; Nbif.n2i=n2i;
end
if nnz(n3)~=0
    Nbif.n3=n3; Nbif.n3i=n3i;
end
if nnz(n4)~=0
    Nbif.n4=n4; Nbif.n4i=n4i;
end
if nnz(n5)~=0
    Nbif.n5=n5; Nbif.n5i=n5i;
end
if nnz(n6)~=0
    Nbif.n6=n6; Nbif.n6i=n6i;
end
if nnz(nElsi)~=0
    Nbif.nEls=nEls; Nbif.nElsi=nElsi;
end
end
end
% artery bifurcation exponent
art=[];
for i=1:length(locA)
    ac=find(locA(:,2)==locA(i,1));
    if isempty(ac)
    else
        rad=tree_a.segment.avrad(ac);
        rad(end+1)=tree_a.segment.avrad(i);
        eqn= 2*max(rad)^x-sum(rad.^x)==0;
%         eqn= tree_a.segment.avrad(i)^x==sum(tree_a.segment.avrad(ac).^x);
        ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        art(end+1)=ad;
    end
    end
end

% vein bifurcation exponent 
arv=[];
for i=1:length(locV)
    ac=find(locV(:,2)==locV(i,1));
    if nnz(ac)~=0
        rad=tree_v.segment.avrad(ac);
        rad(end+1)=tree_v.segment.avrad(i);
        eqn= 2*max(rad)^x-sum(rad.^x)==0;
%         eqn= tree_v.segment.avrad(i)^x==sum(tree_v.segment.avrad(ac).^x);
        ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        arv(end+1)=ad;
    end
    end
end


%% bif exp for each 

ar2=[];
for b=1:length(Art_bif.n2i)
%     rad=tree_a.segment.avrad(Art_bif.n2branches(b,:));
%     rad(end+1)=tree_a.segment.avrad(Art_bif.n2i(b));
%     eqn= 2*max(rad)^x-sum(rad.^x)==0;
    eqn= tree_a.segment.avrad(Art_bif.n2i(b))^x==sum(tree_a.segment.avrad(Art_bif.n2branches(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        ar2(end+1)=ad;
    end
end
ar3=[];
for b=1:length(Art_bif.n3i)
%     rad=tree_a.segment.avrad(Art_bif.n2branches(b,:));
%     rad(end+1)=tree_a.segment.avrad(Art_bif.n2i(b));
%     eqn= 2*max(rad)^x-sum(rad.^x)==0;
    eqn= tree_a.segment.avrad(Art_bif.n3i(b))^x==sum(tree_a.segment.avrad(Art_bif.n3branches(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        ar3(end+1)=ad;
    end
end
a2=[];
for b=1:length(Art_bif.n2i)
    rad=tree_a.segment.avrad(Art_bif.n2branches(b,:));
    rad(end+1)=tree_a.segment.avrad(Art_bif.n2i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;
%     eqn= tree_a.segment.avrad(Art_bif.n2i(b))^x==sum(tree_a.segment.avrad(Art_bif.n2branches(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        a2(end+1)=ad;
    end
end
a3=[];
for b=1:length(Art_bif.n3i)
    rad=tree_a.segment.avrad(Art_bif.n3branches(b,:));
    rad(end+1)=tree_a.segment.avrad(Art_bif.n3i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;
%     eqn= tree_a.segment.avrad(Art_bif.n2i(b))^x==sum(tree_a.segment.avrad(Art_bif.n2branches(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        a3(end+1)=ad;
    end
end


v2=[];
for b=1:length(Vein_bif.n2i)
    rad=tree_v.segment.avrad(Vein_bif.n2(b,:));
    rad(end+1)=tree_v.segment.avrad(Vein_bif.n2i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        v2(end+1)=ad;
    end
end

v3=[];
for b=1:length(Vein_bif.n3i)
    rad=tree_v.segment.avrad(Vein_bif.n3(b,:));
    rad(end+1)=tree_v.segment.avrad(Vein_bif.n3i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        v3(end+1)=ad;
    end
end

v4=[];
for b=1:length(Vein_bif.n4i)
    rad=tree_v.segment.avrad(Vein_bif.n4(b,:));
    rad(end+1)=tree_v.segment.avrad(Vein_bif.n4i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;  
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        v4(end+1)=ad;
    end
end
v569=[];
for b=1:length(Vein_bif.n5i)
    rad=tree_v.segment.avrad(Vein_bif.n5(b,:));
    rad(end+1)=tree_v.segment.avrad(Vein_bif.n5i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;  
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        v569(end+1)=ad;
    end
end
for b=1:length(Vein_bif.n6i)
    rad=tree_v.segment.avrad(Vein_bif.n6(b,:));
    rad(end+1)=tree_v.segment.avrad(Vein_bif.n6i(b));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;  
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        v569(end+1)=ad;
    end
end
    rad=tree_v.segment.avrad(Vein_bif.nEls{2});
    rad(end+1)=tree_v.segment.avrad(Vein_bif.nElsi(2));
    eqn= 2*max(rad)^x-sum(rad.^x)==0;  
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        v569(end+1)=ad;
    end

vr2=[];
for b=1:length(Vein_bif.n2i)
    eqn= tree_v.segment.avrad(Vein_bif.n2i(b))^x==sum(tree_v.segment.avrad(Vein_bif.n2(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        vr2(end+1)=ad;
    end
end
vr3=[];
for b=1:length(Vein_bif.n3i)
    eqn= tree_v.segment.avrad(Vein_bif.n3i(b))^x==sum(tree_v.segment.avrad(Vein_bif.n3(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        vr3(end+1)=ad;
    end
end
vr4=[];
for b=1:length(Vein_bif.n4i)
    eqn= tree_v.segment.avrad(Vein_bif.n4i(b))^x==sum(tree_v.segment.avrad(Vein_bif.n4(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        vr4(end+1)=ad;
    end
end
vr569=[];
for b=1:length(Vein_bif.n5i)
    eqn= tree_v.segment.avrad(Vein_bif.n4i(b))^x==sum(tree_v.segment.avrad(Vein_bif.n4(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        vr569(end+1)=ad;
    end
end
for b=1:length(Vein_bif.n6i)
    eqn= tree_v.segment.avrad(Vein_bif.n4i(b))^x==sum(tree_v.segment.avrad(Vein_bif.n4(b,:)).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        vr569(end+1)=ad;
    end
end
    eqn= tree_v.segment.avrad(Vein_bif.nElsi(2))^x==sum(tree_v.segment.avrad(Vein_bif.nEls{2}).^x);
    ad = vpasolve(eqn,x);
    if isempty(ad)
    else
        vr569(end+1)=ad;
    end

% Degree of asymmetric

ratio=[];
for b=1:length(Art_bif.n2i)
    rad=tree_a.segment.avrad(Art_bif.n2branches(b,:));
    ratio(end+1)=min(rad)/max(rad);
end

vratio=[];
for b=1:length(Vein_bif.n2i)
    rad=tree_v.segment.avrad(Vein_bif.n2(b,:));
    vratio(end+1)=min(rad)/max(rad);
end

Lratio=[];
for b=1:length(Art_bif.n2i)
    rad=tree_a.segment.L(Art_bif.n2branches(b,:));
    Lratio(end+1)=min(rad)/max(rad);
end

vLratio=[];
for b=1:length(Vein_bif.n2i)
    rad=tree_v.segment.L(Vein_bif.n2(b,:));
    vLratio(end+1)=min(rad)/max(rad);
end