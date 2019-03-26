f = fopen('plot.txt');
data = textscan(f,'%f %f %f %f %f %f');
fclose(f);
t = 0.1*(1:length(data{1}));
plot(t,data{4},t,data{5},t,data{6});
legend('rightward','upward','leftward');
xlabel('t(s)');
ylabel('v(m/s)');
title('三队车速度曲线');