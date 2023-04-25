%%更新画图
%测试集测试
% vaild
clc,clear,close all
T=readtable("data1_in_.csv");
data1=[];
M=[]
for i =1:50
    s=17189+(i-1)*50;
    e=min(19701,s+50);
    temp=T(s:e,:);
    mu=temp(:,5);
    m=mean(table2array(mu)');
    M=[M;m];
    a=array2table((1:e-s+1)');
    int=e-s+1;
    idx=i*ones(int,1);
    idx=array2table(idx)
    temp=[idx,a,temp];
    data1=[data1;temp ];
end
data2=[];
M2=[];
for i2=1:50
    s2=2906+(i2-1)*120;
    e2=min(8886,s2+120);
    temp2=T(s2:e2,:);
    mu2=temp2(:,5);
    m2=mean(table2array(mu2)');
    M2=[M2;m2];
    a2=array2table((1:e2-s2+1)');
    int2=e2-s2+1;
    idx2=(i2+50)*ones(int2,1);
    idx2=array2table(idx2);
    temp2=[idx2,a2,temp2];
    data2=[data2;temp2];
end
data2=renamevars(data2,"idx2","idx");
data=[data1;data2];
writetable(data,"data1_testset.txt");
M0=[M;M2];
M0=array2table(M0);
writetable(M0,"data1_psi.txt")