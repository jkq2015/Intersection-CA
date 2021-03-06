f = fopen('plot2.txt');
data = textscan(f,'%f %f %f %f');
fclose(f);
t = 0.1*(1:length(data{1}));
figure;
plot(t,data{3},t,data{4},'LineWidth',1);
h = legend('车队1','车队2','Location','NorthWest');
h.FontSize = 18;
xlabel('t/s','FontSize',18);
ylabel('v/(m\cdots^{-1})','FontSize',18);
title('两车队头车速度曲线','FontSize',18);
% saveas(1,'two_cars.emf');

f = fopen('plot3.txt');
data = textscan(f,'%f %f %f %f %f %f');
fclose(f);
t = 0.1*(1:length(data{1}));
figure;
plot(t,data{4},t,data{5},t,data{6},'LineWidth',1);
h = legend('车队1','车队2','车队3');
h.FontSize = 18;
xlabel('t/s','FontSize',18);
ylabel('v/(m\cdots^{-1})','FontSize',18);
title('三车队头车速度曲线','FontSize',18);
% saveas(1,'three_cars.emf');