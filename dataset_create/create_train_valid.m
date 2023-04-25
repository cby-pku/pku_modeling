%%
%data1_trainset数据构建
clc,clear,close all
T=readtable("data1_in_.csv");
data1=[];
Mt=[]
for i =1:5
    s=10129+(i-1)*226;
    e=min(10129+i*226,11257);
    temp=T(s:e,:);
    mu=temp(:,5);
    m=mean(table2array(mu)');
    Mt=[Mt;m];
    a=array2table((1:e-s+1)')
    int=e-s+1
    idx=i*ones(int,1);
    idx=array2table(idx)
    temp=[idx,a,temp];
    data1=[data1;temp ];
end
%对于每个cycle，其rui值应该标注为平均值
data2=[];
for i2=6-5:10-5
    s2=17771+(i2-1)*243;
    e2=min(18988,17771+i2*243);
    temp2=T(s2:e2,:);
    int2=e2-s2+1
    if int2<100
        break
    end
    mu=temp2(:,5);
    m=mean(table2array(mu)');
    Mt=[Mt;m];
    a2=array2table((1:e2-s2+1)')
    idx2=(i2+5)*ones(int2,1);
    idx2=array2table(idx2);%这样才能保证上下变量名idx2和a2的不重复
    temp2=[idx2,a2,temp2];
    data2=[data2;temp2];
end
%203
data3=[];
for i3=1:6
    s3=955+(i3-1)*203;
    e3=min(s3+203,2174);
    temp3=T(s3:e3,:);
    int3=e3-s3+1
    if int3<100
        break
    end
        mu=temp3(:,5);
    m=mean(table2array(mu)');
    Mt=[Mt;m];
    a3=array2table((1:int3)');
    idx3=(i3+10)*ones(int3,1);
    idx3=array2table(idx3);
    temp3=[idx3,a3,temp3];
    data3=[data3;temp3];
end
%到16个了
data4=[]
for i4=1:30
    %210
    s4=2903+(i4-1)*210;
    e4=min(9211,s4+210);
    temp4=T(s4:e4,:)
    int4=e4-s4+1;
    if int4<100
        break
    end
        mu=temp4(:,5);
    m=mean(table2array(mu)');
    Mt=[Mt;m];
    a4=array2table((1:int4)');
    idx4=(i4+16)*ones(int4,1);
    idx4=array2table(idx4);
    temp4=[idx4,a4,temp4];
    data4=[data4;temp4];
end
%到46个了
data5=[]
for i5=1:16
    %220
    s5=5595+(i5-1)*220;
    e5=min(8875,s5+220);
    temp5=T(s5:e5,:);
    int5=e5-s5+1;
    if int5<100
        break
    end
        mu=temp5(:,5);
    m=mean(table2array(mu)');
    Mt=[Mt;m];
    a5=array2table((1:int5)');
    idx5=(i5+46)*ones(int5,1);
    idx5=array2table(idx5);
    temp5=[idx5,a5,temp5];
    data5=[data5;temp5];
end

data2=renamevars(data2,"idx2","idx");
data3=renamevars(data3,"idx3","idx");
data4=renamevars(data4,"idx4","idx");
data5=renamevars(data5, "idx5","idx");
data=[data1;data2;data3;data4;data5];
writetable(data,"data1_trainset.txt");
Mt=array2table(Mt);
writetable(Mt,"train_psi.txt");


%%
% vaild数据集构建
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


