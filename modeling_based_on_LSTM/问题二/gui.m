% 诡集数据集构建
%用于预警阈值确定
clc,clear,close all
T=readtable("data1_in_.csv");
data=[]
M=[]
%100个cycle，10个一步
for i=1:100
    s=16211+(i-1)*8;
    e=min(17034,s+8);
    idx=i*ones(e-s+1,1);
    idx=array2table(idx);
    temp=T(s:e,:);
    m=mean(table2array(temp(:,5))');
    M=[M;m];
    int =e-s+1;
    a=array2table((1:int)');
    ans=[idx,a,temp];
    data=[data;ans];
end
M(end,:)=[];
M=array2table(M);
writetable(M,"data1_g3_psi.txt");
writetable(data,"data1_g3_test.txt");