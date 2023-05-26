Nsample = 200;
NumberIterations = size(Envdata.time, 2);

% N_min, T_R1, k_A, N_max, J_max, alpha, I_sat, k_C

%Parameters
A_O = zeros(Nsample,1) + 6;
alpha = zeros(Nsample,1) + 3.75*10^(-5);
C_min = zeros(Nsample,1) + 0.01;
C_struct = zeros(Nsample,1) + 0.20;
gamma = zeros(Nsample,1) + 0.5;
epsilon = zeros(Nsample,1) + 0.22;
I_sat = zeros(Nsample,1)+ 200;
J_max = zeros(Nsample,1) + 1.4*10^(-4);
k_A = zeros(Nsample,1) + 0.03;
k_dw = zeros(Nsample,1) + 0.0785;
k_C = zeros(Nsample,1) + 2.1213;
k_N = zeros(Nsample,1) + 2.72;
m_1 = zeros(Nsample,1) + 0.1085;
m_2 = zeros(Nsample,1) + 0.03;
my_max = zeros(Nsample,1) + 0.18;
N_min = zeros(Nsample,1) + 0.01;
N_max = zeros(Nsample,1) + 0.022;
N_struct = zeros(Nsample,1) + 0.01;
P_1 = zeros(Nsample,1) + 1.22*10^(-3);
P_2 = zeros(Nsample,1) + 1.44*10^(-3);
a_1 = zeros(Nsample,1) + 0.85;
a_2 = zeros(Nsample,1) + 0.3;
R_1 = zeros(Nsample,1) + 2.2*10^(-4); 
R_2 = zeros(Nsample,1) + 5.429*10^(-4);
T_R1 = 285;
T_R2 = 290;
T_AP = 1694.4;
T_APH = 25924;
T_APL = 27774;
T_AR = 11033;
U_065 = zeros(Nsample,1) + 0.03;
K_X = zeros(Nsample,1) + 4;

N_min_true = zeros(1,NumberIterations) + N_min(1);
N_min_non = zeros(Nsample,1) + N_min(1);
alpha_true = zeros(1,NumberIterations) + alpha(1);
k_A_true = zeros(1,NumberIterations) + k_A(1);
J_max_true = zeros(1,NumberIterations) + J_max(1);
I_sat_true = zeros(1,NumberIterations) + I_sat(1);
k_C_true = zeros(1,NumberIterations) + k_C(1);
% N_min_true (2000:4000) = N_min(1) + randn * N_min(1)*0.3;
% N_min_true(4000:6000) = N_min(1) + randn * N_min(1)*0.3;
% N_min_true(6000:8000) = N_min(1) + randn * N_min(1)*0.3;
% N_min_true(8000:end) = N_min(1) + randn * N_min(1)*0.3;

% m_1_true = zeros(1,NumberIterations) + m_1(1);
% m_1_true(4000:end) = m_1(1)*2;
% A_O = 8; 
% alpha = 1.204*10^(-5); 
% C_min = 0.01; 
% C_struct = 0.20; 
% gamma = 0.5; 
% epsilon = 0.22; 
% I_sat = 90; 
% J_max = 2.8*10^(-5); 
% k_A = 0.027;
% k_dw = 0.0685; 
% k_C = 2.1213; 
% k_N = 2.8929; 
% m_1 = 0.1085; 
% m_2 = 0.03; 
% my_max = 0.1823;
% N_min = 0.0088; 
% N_max = 0.0216; 
% N_struct = 0.0121; 
% P_1 = 1.44*10^(-3); 
% P_2 = 1.44*10^(-3); 
% a_1 = 0.85; %
% a_2 = 0.3; %
% R_1 = 2.2*10^(-4); 
% R_2 = 5.429*10^(-4); 
% T_R1 = 285; 
% T_R2 = 290; 
% T_AP = 1694.4; 
% T_APH = 25924; 
% T_APL = 27774; 
% T_AR = 6200; 
% U_065 = 0.03; 
% K_X = 56; 

%% Temperature Dataset
t1 = datetime(Envdata.time(1,1),Envdata.time(2,1),Envdata.time(3,1),Envdata.time(4,1),Envdata.time(5,1),Envdata.time(6,1), 'Format','yyyy-MM-dd HH:mm:ss');
t2 = datetime(Envdata.time(1,end),Envdata.time(2,end),Envdata.time(3,end),Envdata.time(4,end),Envdata.time(5,end),Envdata.time(6,end), 'Format','yyyy-MM-dd HH:mm:ss');
time = t1:minutes(15):t2;

%% Initial values
A0 = [startareas(:,1); startareas(:,2); startareas(:,3); startareas(:,4); startareas(:,5)];
[A0_std, A0_mu] = std(A0(:, 1));

N_0 = zeros(Nsample, 1) + 0.01;
C_0 = zeros(Nsample, 1) + 0.05;
A_0 = zeros(Nsample, 1) + A0_mu/100; 
%A_0 = [A0; A0; A0; A0]/100;

N_min_0 = 0.01;
T_R1_0 = 285;
k_A_0 = 0.6;
N_max_0 = 0.022;
J_max_0 = 1.4 * 10^(-4);
alpha_0 = 3.75 * 10^(-5);
I_sat_0 = 200;
k_C_0 = 2.1213;


%% Statevariable
X_T = zeros(Nsample, NumberIterations);
X_U = zeros(Nsample, NumberIterations);
X_XNO3 = zeros(Nsample, NumberIterations);
X_I = zeros(Nsample, NumberIterations);
pert = zeros(Nsample, NumberIterations);
pert_I = zeros(Nsample, NumberIterations);
pert_NO3 = zeros(Nsample, NumberIterations);
pert_U = zeros(Nsample, NumberIterations);
X_XNO3_true = zeros(1, NumberIterations);
X_I_true = zeros(1, NumberIterations);

for n = 1:NumberIterations
    X_T(:, n) = Envdata.T(:,:,2,n) ;
    X_U(:, n) = 0.06;
    X_XNO3(:, n) = Envdata.NO3(:,:,2,n);
    X_I(:, n) = Envdata.PAR(:,:,2,n);
end

%% Standard variation and mean
[T_std, T_mu] = std(X_T(1, :));
[U_std, U_mu] = std(X_U(1, :));
[NO3_std, NO3_mu] = std(X_XNO3(1, :));
[I_std, I_mu] = std(X_I(1, :));

%% Measurements
A_measurements = zeros(50,6);
t_m_1 = datetime(SESdata2018.dates(1), 'Format', 'yyyy-MM-dd HH:mm:ss.S', 'convertFrom', 'datenum');
t_m_2 = datetime(SESdata2018.dates(end), 'Format', 'yyyy-MM-dd HH:mm:ss.S', 'convertFrom', 'datenum');
time_measurements = linspace(t_m_1, t_m_2, size(SESdata2018.dates,2));
time_measurements_number = zeros(1, size(SESdata2018.dates,2));
for i = 1:size(SESdata2018.dates, 2)
    time_measurements(1,i) = datetime(SESdata2018.dates(i), 'Format', 'yyyy-MM-dd HH:mm:ss.S', 'convertFrom', 'datenum');
    A_measurements(:,i) = reshape(SESdata2018.FrondAreas{i}'/100, 1, []);
    time_measurements_number(i) = getMeasurements(SESdata2018.dates(i), NumberIterations, Envdata);
end

%% Randomize state variable
for i = 1:Nsample
     for j = 1:NumberIterations
          pert(i,j+1) = gaussMarkov(pert(i,j), 0.8, 15/1440, T_mu*0.2, 0);
          X_T(i, j) = X_T(i, j) + pert(i,j);
          pert_I(i,j+1) = gaussMarkov(pert_I(i,j), 0.8, 15/1440, I_mu*0.2, 0);
          X_I(i, j) = X_I(i, j) + pert_I(i,j);
          if (X_I(i,j) < 0)
              X_I(i,j) = X_I(i,j-1);
          end
          pert_NO3(i,j+1) = gaussMarkov(pert_NO3(i,j), 0.8, 15/1440, NO3_mu*0.2, 0);
          X_XNO3(i, j) = X_XNO3(i, j) + pert_NO3(i,j);
          if (X_XNO3(i,j) < 0)
              X_XNO3(i,j) = X_XNO3(i,j-1);
          end
%           X_XNO3_true(j) = X_XNO3(1,j) + X_XNO3(1,j)*0.3;
%           X_I_true(j) = X_I(1,j) + X_I(1,j)*0.3;

%           pert_U(i,j+1) = gaussMarkov(pert_U(i,j), 0.8, 15/1440, U_mu*0.4, 0);
%           X_U(i, j) = X_U(i, j) + pert_U(i,j);
%           if (X_U(i,j) < 0)
%               X_U(i,j) = X_U(i,j-1);
%           end
      end
end



% N_min_true = zeros(1, NumberIterations) + N_min(1);
% k_A_true = zeros(1, NumberIterations) + k_A(1);
% 
% for i = 1:NumberIterations
%     N_min_true(i) = addRandomAddition(N_min_true(i), N_min(1)*0.1);
%     k_A_true(i) = addRandomAddition(k_A_true(i), k_A(1)*0.1);
% end

%% Perturbate parameters
% N_min, k_A, N_max, J_max, alpha, I_sat, k_C

for i = 2:Nsample
%     alpha(i) = addRandomAddition(alpha(i), alpha(i)*0.2);
%     k_A(i) = addRandomAddition(k_A(i), k_A(i)*0.2);
%     J_max(i) = addRandomAddition(J_max(i), J_max(i)*0.2);
%     I_sat(i) = addRandomAddition(I_sat(i), I_sat(i)*0.2);
%     k_C(i) = addRandomAddition(k_C(i), k_C(i)*0.2);
    N_min(i) = addRandomAddition(N_min(i), N_min(i)*0.2);
    if (N_min(i) > N_max(1))
        N_min(i) = N_min(i-1);
    end
%       m_1(i) = addRandomAddition(m_1(i), m_1(i)*0.2);
%     N_max(i) = addRandomAddition(N_max(i), N_max(i)*0.2);
% %     while N_max(i) <= N_min(i)
% %         N_max(i) = addRandomAddition(N_max(i), N_max(i)*0.2);
% %     end
end

%% Perturbate inital value
for i = 2:Nsample
    A_0(i) = addRandomAddition(A_0(i), A0_mu/100*0.2);
    N_0(i) = addRandomAddition(N_0(i), N_0(i)*0.2);
    C_0(i) = addRandomAddition(C_0(i), C_0(i)*0.2);
end

%% Gauss Markov noise
function x_next = gaussMarkov(x, beta, dt, sigma, mean)
f = exp(-beta*dt);
r = randn*sigma + mean;
x_next = f*x + sqrt(1-f^2)*r;
end

%% Random noise for parameters
function y = addRandomAddition(x, sigma)
y = x + randn*sigma;
if (y)<0 
    y = 0;
end
end

%% Get measurements
function data = getMeasurements(date, NumberIterations, Envdata)
    [Y, M, D, H, MS, S] = datevec(date);
    for i = 1:NumberIterations
        if ((Envdata.time(1,i) == Y) && (Envdata.time(2, i) == M) && (Envdata.time(3,i) == D) && (Envdata.time(4,i) == H) && (Envdata.time(5,i) == MS) && (Envdata.time(6,i) == S) )
            data = i;
        end
    end
end