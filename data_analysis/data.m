clc,clear,close all;
%样本提取后进行相关分析
T=readtable("data_corrected_1.csv");
%钻深
bit_depth=table2array(T(:,2));
%钻压
WOB=table2array(T(:,3));
%井眼深度
hole_depth=table2array(T(:,4));
%流速
f=table2array(T(:,5));
%立管压力
psi=table2array(T(:,6));
%扭矩
torque=table2array(T(:,7));
%转速
w=table2array(T(:,8));
intex=find(f==0&abs(bit_depth-hole_depth)>50);
intex=unique(intex)
x=bit_depth(intex,1);
y=psi(intex,1);
b=regress(y,[ones(length(x),1) x]);
plot(x,y,'o');
hold on;
plot(x,[ones(length(x),1) x]*b,'-');
xlabel("bit_depth");
ylabel("psi");
saveas(gcf,"bit_depth-psi-data1.png");
