%%
%数据预处理
%数据异常值剔除
T4=readtable("TrainingData2.csv");

LP4=table2array(T4(1:200:end,6));
plot(LP4,"LineWidth",2);
xlabel("time(s)");
ylabel("psi");
saveas(gcf,"raw_psi-data2.png");

quartiles_x=quantile(LP4,[0.25,0.5,0.75]);
iqr=quartiles_x(3)-quartiles_x(1);
lowerbound=quartiles_x(1)-1.5*iqr;
upperbound=quartiles_x(3)+1.5*iqr;
new_lp4=LP4(find(LP4>lowerbound&LP4<upperbound),:);
u4=mean(new_lp4);
std4=std(new_lp4);
idx=abs(new_lp4-u4)>3*std4;
new_lp4(idx)=[];
data_delete=T4(idx,:);
T4(idx,:)=[];
writetable(data_delete,"data_delet" + ...
    "e_2.csv");
writetable(T4,"data_corrected_2.csv");



plot(new_lp4,"LineWidth",2);
xlabel("time(s)");
ylabel("psi");
saveas(gcf,"correct_psi-data2.png");

%%
% 修正列
T=readtable("data_corrected_4.csv");
[T(:,7),T(:,8)]=deal(T(:,8),T(:,7));
writetable(T,"data_corrected_4.csv");