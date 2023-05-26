timestampCorrection = 5377;
timestampMeasurement = 5377;
length = 390;
height = 250;
heightErrorSignal = 250;
positionGraph = 750;
positionMean = 415;
postionErrorSignal = 80;
positionHistogram1 = 750;
positionHistogram2 = 415;
positionHistogram3 = 80;

%7,5 -> 28
% 60
% 

a = figure(20);
a.Position = [0 positionGraph length height];
grid on;
title('Area')
ylabel('dm^2')
plot(time, X_a_A)
hold on; 
plot(time, x_t_true(1,:), 'LineWidth',5)
hold off;

figure(30);
grid on;
title('Area')
ylabel('dm^2')
plot(time, x_t_true(1,:) - mean(X_a_A), 'LineWidth', 2)
hold on; 
xline(time(96*7*2))
xline(time(96*7*2*2))
xline(time(96*7*2*3))
xline(time(96*7*2*4))
xline(time(96*7*2*5))
xline(time(96*7*2*6))
xline(time(96*7*2*7))
hold off;

b = figure(21);
b.Position = [length*1 positionGraph length height];
plot(time(1:end-1), X_a_N(2:end-1, 1:end-1))
grid on;
title('Nitrogen')
ylabel('gN(gsw)^{-1}');
hold on; 
plot(time(1:end-1), x_t_true(2, 1:end-1), 'LineWidth', 3);
hold off;
% 
c = figure(22);
c.Position = [length*2 positionGraph length height];
plot(time(1:end-1), X_a_C(2:end-1, 1:end-1))
grid on;
title('Carbon')
ylabel('gC(gsw)^{-1}');
hold on; 
plot(time(1:end-1), x_t_true(3, 1:end-1), 'LineWidth', 3);
hold off;

d = figure(23);
d.Position = [length*0 positionMean length height];
plot(time(1:end-1), mean(X_a_A(2:end-1, 1:end-1)))
grid on;
title('True process AREA vs mean value X_a')
ylabel('dm^2')
hold on; 
plot(time(1:end-1), x_t_true(1, 1:end-1), 'LineWidth', 3);
hold off;

t = figure(40);
t.Position = [length*0 positionMean length height];
plot(time(1:end-1), Y_A(2:end-1, 1:end-1))
grid on;
title('Non correction')
ylabel('dm^2')
hold on; 
plot(time(1:end-1), x_t_true(1, 1:end-1), 'LineWidth', 3);
hold off;

e = figure(24);
e.Position = [length*1 positionMean length height];
plot(time(1:end-1), mean(X_a_N(2:end-1, 1:end-1)))
grid on;
title('True process NITROGEN vs mean value X_a')
ylabel('gN(gsw)^{-1}');
hold on; 
plot(time(1:end-1), x_t_true(2, 1:end-1), 'LineWidth', 3);
hold off;

u = figure(41);
u.Position = [length*1 positionMean length height];
plot(time(1:end-1), Y_N(2:end-1, 1:end-1))
grid on;
title('Non correction')
ylabel('dm^2')
hold on; 
plot(time(1:end-1), x_t_true(2, 1:end-1), 'LineWidth', 3);
hold off;

f = figure(25);
f.Position = [length*2 positionMean length height];
plot(time(1:end-1), mean(X_a_C(2:end-1, 1:end-1)))
grid on;
title('True process CARBON vs mean value X_a')
ylabel('gC(gsw)^{-1}');
hold on; 
plot(time(1:end-1), x_t_true(3, 1:end-1), 'LineWidth', 3);
hold off;

v = figure(42);
v.Position = [length*2 positionMean length height];
plot(time(1:end-1), Y_C(2:end-1, 1:end-1))
grid on;
title('Non correction')
ylabel('dm^2')
hold on; 
plot(time(1:end-1), x_t_true(3, 1:end-1), 'LineWidth', 3);
hold off;

g = figure(26);
g.Position = [length*3 positionHistogram1 length height];
histogram(X_f_A(:, timestampCorrection));
grid on;
hold on;
histogram(X_a_A(:, timestampCorrection), 'FaceColor', "#77AC30");
hold on;
xline(x_t_true(1,timestampCorrection), 'LineWidth', 2)
legend('X_f (forecast)', 'X_a (analysis)', 'x_t (true)')
title('Area');
hold off; 

h = figure(27);
h.Position = [length*4 positionHistogram1 length height];
histogram(X_f_N(:, timestampCorrection));
grid on;
hold on;
histogram(X_a_N(:, timestampCorrection), 'FaceColor', "#77AC30");
hold on;
xline(x_t_true(2,timestampCorrection), 'LineWidth', 2)
legend('X_f (forecast)', 'X_a (analysis)', 'x_t (true)')
title('Nitrogen');
hold off; 

i = figure(28);
i.Position = [length*3 positionHistogram2 length height];
histogram(X_f_C(:, timestampCorrection));
grid on;
hold on;
histogram(X_a_C(:, timestampCorrection), 'FaceColor', "#77AC30");
hold on;
xline(x_t_true(3,timestampCorrection), 'LineWidth', 2)
legend('X_f (forecast)', 'X_a (analysis)', 'x_t (true)')
title('Carbon');
hold off; 

% j = figure(29);
% j.Position = [length*4 positionHistogram2 length height];
% histogram(D_A(:, timestampMeasurement));
% grid on;
% hold on;
% xline(d_true(1, timestampMeasurement), 'LineWidth', 2)
% title('D Area');
% hold off; 

k = figure(31);
clf
k.Position = [length*3 positionHistogram3 length height];
% shadedErrorBar(1:9505, X_a_Nmin,{@mean,@std});
hold on; 
plot(1:9505, N_min_true)
title('k_{A}')
grid on;
set(gca,'XTickLabel',datevec(time))
hold off;

% k = figure(31)
% k.Position = [length*4 positionHistogram3 length height];
% plot(time, pert_param)
% hold on; 
% plot(time, m_1_true, 'LineWidth', 3)
% title('m_1')
% grid on;
% hold off;

% if(size(M,1) == 2)
%     k = figure(30);
%     k.Position = [length*3 positionHistogram3 length height];
%     histogram(D_N(:, timestampMeasurement));
%     grid on;
%     hold on;
%     xline(d_true(2, timestampMeasurement), 'LineWidth', 2)
%     title('D Nitrogen');
%     hold off; 
% end

% if(size(M,1) == 3)
%     k = figure(30);
%     k.Position = [length*3 positionHistogram3 length height];
%     histogram(D_N(:, timestampMeasurement));
%     grid on;
%     hold on;
%     xline(d_true(2, timestampMeasurement), 'LineWidth', 2)
%     title('D Nitrogen');
%     hold off; 
% 
%     l = figure(31);
%     l.Position = [length*4 positionHistogram3 length height];
%     histogram(D_C(:, timestampMeasurement));
%     grid on;
%     hold on;
%     xline(d_true(3, timestampMeasurement), 'LineWidth', 2)
%     title('D Carbon');
%     hold off; 
% end

% m = figure(32);
% plot(time, D_A(1,:));
% grid on;
% hold on;
% plot(time, X_a_A(1,:));
% plot(time, X_f_A(1,:));
% hold off;

% n = figure(33);
% n.Position = [0 postionErrorSignal length heightErrorSignal];
% plot(time, x_t_true(1, :) - mean(X_a_A));
% grid on;
% title('Error signal - difference true and estimated value area')
% ylabel('dm^2')
% 
% o = figure(34);
% o.Position = [length*1 postionErrorSignal length height];
% plot(time, x_t_true(2, :) - mean(X_a_N));
% grid on;
% title('Error signal - difference true and estimated value nitrogen')
% 
% p = figure(35);
% p.Position = [length*2 postionErrorSignal length height];
% plot(time, x_t_true(3, :) - mean(X_a_C));
% grid on;
% title('Error signal - difference true and estimated value carbon')

% q = figure(36);
% histfit(X_f_A(:, timestamp));
% hold on;
% histfit(X_a_A(:, timestamp));
% hold off;

% r = figure(37);
% plot(time(1:end-1), x_t_true(4, 1:end-1) - mean(X_a_N_min(2:end-1, 1:end-1)))
% grid on;