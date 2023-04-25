clc,clear,close all;
T1=readtable("TestData1(1).csv");
T2=readtable("TestData2.csv");
%时间
time1=T1(:,1);
time2=T2(:,1);
%流速
flow1=T1(:,2);
flow2=T2(:,2);
%立管压力
psi1=T1(:,3);
psi2=T2(:,3);
%井眼深度
zbd1=T1(:,4);
zbd2=T2(:,4);
%钻头深度
z1=T1(:,5);
z2=T2(:,5);
%扭矩
torque1=T1(:,6);
torque2=T2(:,6);
%转速
w1=T1(:,7);
w2=T2(:,7);
%WOB钻压
WOB1=T1(:,8);
WOB2=T2(:,8);
tmp1=array2table(ones(49767,2));
tmp2=array2table(ones(69601,2));
raw1=[z1 WOB1 zbd1 flow1 psi1 w1 torque1 tmp1];
raw2=[z2 WOB2 zbd2 flow2 psi2 w2 torque2 tmp2];
traindata1=raw1(1:35000,:);
M1=[];
testdata1=raw1(35000:end,:);
trainset=[]
max4=23759281.8740341;
for i =1:250
   s=1+(i-1)*10;
   e=min(35000,s+10);
   int =e-s+1;
   mu=traindata1(s:e,5);
   m=mean(table2array(mu)');
   M1=[M1;m];
   idx=i*ones(int,1);
   idx=array2table(idx);
   a=((1:int)');
   a=array2table(a);
   temp=[idx,a,traindata1(s:e,:)];
   trainset=[trainset;temp];
end
%得到trainset;
M1=array2table(M1);
writetable(M1,"Test1_1_psi.txt");%标注，既可以做
writetable(trainset,"Test1_1_set.txt");%数据集