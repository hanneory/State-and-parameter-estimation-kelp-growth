%% This file is used for the Twin experiment

%% Parameterer man vil tilpasse:
% N_min, T_R1, k_A, N_max, J_max, alpha, I_sat, k_C
h = 15/1440;
correctionStep = 10;

% Denne kan eg variere  
%M = A, N, C, N_min, k_A
%M = [1 0 0];
%M = [1 0 0; 0 1 0];
M = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0];
%M = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
%M = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1];

% Denne må eg tilpasse til M
%measStd = 0.5;
%measStd = 0.01;
measStd = [0.5; 0.005; 0.01];
%measStd = [0.5; 0.01; 0.01; 0.01; 0.01];
%measStd = [0.5; 0.01];
R = measStd.^2;
Nstates = size(measStd,1);

Y_A = zeros(Nsample, NumberIterations);
Y_N = zeros(Nsample, NumberIterations);
Y_C = zeros(Nsample, NumberIterations);
C_content = zeros(Nsample, NumberIterations);
N_content = zeros(Nsample, NumberIterations);
netCarbonFixed = zeros(Nsample, NumberIterations);
grossFrond = zeros(Nsample, NumberIterations);
Y_A(:,1) = A_0;
Y_N(:,1) = N_0;
Y_C(:,1) = C_0;
netCarbonFixed(:,1) = C_0;
grossFrond(:,1) = A_0;
EnKF_A = zeros(Nsample, NumberIterations);
EnKF_N = zeros(Nsample, NumberIterations);
EnKF_C = zeros(Nsample, NumberIterations);
%x_t_true = zeros(size(measStd,1), NumberIterations);
x_t_true = zeros(3, NumberIterations);
%x_t_true(:,1) = [A_0(1); N_0(1); C_0(1); N_min(1); k_A(1)];
x_t_true(:,1) = [A_0(1); N_0(1); C_0(1)];
%d_true = zeros(size(measStd,1), NumberIterations);
d_true = zeros(3, NumberIterations);

D_A = zeros(Nsample, NumberIterations);
D_N = zeros(Nsample, NumberIterations);
D_C = zeros(Nsample, NumberIterations);
D_N_min = zeros(Nsample, NumberIterations);

X_a_A = zeros(Nsample, NumberIterations);
X_a_N = zeros(Nsample, NumberIterations);
X_a_C = zeros(Nsample, NumberIterations);
X_a_N_min = zeros(Nsample, NumberIterations);
X_a_kA = zeros(Nsample, NumberIterations);

X_f_A = zeros(Nsample, NumberIterations);
X_f_N = zeros(Nsample, NumberIterations);
X_f_C = zeros(Nsample, NumberIterations);
X_f_N_min = zeros(Nsample, NumberIterations);
X_f_kA = zeros(Nsample, NumberIterations);

X_a_A(:,1) = A_0;
X_a_N(:,1) = N_0;
X_a_C(:,1) = C_0;
X_a_N_min(:,1) = N_min;
X_a_kA(:,1) = k_A;

% Kalman filter
% X_kf_A = zeros(Nsample, NumberIterations);
% X_kf_N = zeros(Nsample, NumberIterations);
% X_kf_C = zeros(Nsample, NumberIterations);

Kalman_gain_matrix = zeros(size(measStd,1));
D_matrix = zeros(size(measStd,1), Nsample);
%cov_XF_matrix = zeros(size(measStd,1));
cov_XF_matrix = zeros(5);

global index
index = 1;

for n = 1:(NumberIterations-1)
    D = zeros(size(measStd,1), Nsample);
    A = X_a_A(:,n);
    N = X_a_N(:,n);
    C = X_a_C(:,n);
    T = X_T(:, n);
    U = X_U(:,n);
    NO3 = X_XNO3(:,n);
    I = X_I(:,n);
   
    [A_dot_non, N_dot_non, C_dot_non, ~, ~, ~, ~, ~, ~, ~] = kelp(N_struct, C_struct, k_A, k_N, k_C, k_dw, N_min, C_min, m_1, m_2, A_O, epsilon, K_X, N_max, J_max, U_065, R_1, T_AR, T_R1, gamma, alpha, I_sat, P_1, T_AP, T_APL,T_APH, U, NO3, T, I, Y_N(:,n), Y_C(:,n), Y_A(:,n));
  
    Y_A(:, n+1) = Y_A(:, n) + A_dot_non.*h;
    Y_N(:, n+1) = Y_N(:, n) + N_dot_non.*h;
    Y_C(:, n+1) = Y_C(:, n) + C_dot_non.*h;
   
%     X_a = [X_a_A(:, n)'; X_a_N(:, n)'; X_a_C(:, n)'; X_a_N_min(:,n)'; X_a_kA(:,n)'];
    %X_a = [X_a_A(:, n)'; X_a_N(:, n)'; X_a_C(:, n)'];

    % Integrate "true" process
%     x_t = [x_t_true(1,n) + (A_dot_true(1)*h*1);
%            x_t_true(2,n) + (N_dot_true(1)*h*1);
%            x_t_true(3,n) + (C_dot_true(1)*h*1);
%            x_t_true(4,n);
%            x_t_true(5,n)];
    
    %x_t = [x_t_true(1,n) + (A_dot_true(1)*h*1);
    %       x_t_true(2,n) + (N_dot_true(1)*h*1);
    %       x_t_true(3,n) + (C_dot_true(1)*h*1)];

    % Calculate forecast value
%     X_f = [(X_a(1, :)' + A_dot.*h)';
%            (X_a(2, :)' + N_dot.*h)';
%            (X_a(3, :)' + C_dot.*h)';
%            X_a(4, :);
%            X_a(5,:)];


    %X_f = (X_a(1, :)' + A_dot.*h)';
% 
%     covXF = cov(X_f');
%     cov_XF_matrix(:,:, n) = covXF;

    
%     if (rem(n, correctionStep) == 0)
%         
%         % Analysis step. Below measurement without error.
%         d = M * x_t;
%     
%         % Set up measurement ensemble D with errors added:
%         for j = 1:Nsample
%             D(:,j) = d + measStd .* randn(size(measStd));
%         end
%     
%         errorCovarianceMatrix = cov(D');
%         %C_ee = diag(measStd);
%         if(size(M,1) > 1)
%             C_ee = cov(D');
%         else
%             C_ee = diag(measStd);
%         end
%     
%         X_f
%         MX = M*X_f;
%         MA = MX - (1/(Nsample))*(MX*ones(Nsample,1))*ones(1,Nsample);
%         P = (1/(Nsample-1)).*MA*(MA') + C_ee;
%         %if(rank(P) ~= size(P,2))
%         %    disp('P not full rank')
%         %end
%     
%         A_k = X_f - (1/Nsample)*X_f*ones(Nsample,1)*ones(1,Nsample);
%     
%         % Compute analysis:
%         X_a_2 = X_f + (1/(1+Nsample))*A_k*(MA')*(inv(P))*(D - MX);
%     
%         X_a_A(:, n+1) = X_a_2(1,:)';
%         X_a_N(:, n+1) = X_a_2(2,:)';
%         X_a_C(:, n+1) = X_a_2(3,:)';
%         X_a_N_min(:, n+1) = X_a_2(4,:)';
%         X_a_kA(:, n+1) = X_a_2(5,:)';
%         %X_a_N(:, n+1) = X_a_N(:, n) + N_dot.*h;
%         %X_a_C(:, n+1) = X_a_C(:, n) + C_dot.*h;
%         
%         K_gain = (1/(1+Nsample))*A_k*(MA')*(inv(P));
%         Kalman_gain_matrix(:,:, n) = K_gain;
% 
%         %if(rank(K_gain) ~= size(K_gain,2))
%         %    disp("K_gain not full rank");
%         %end
%     else
%         X_a_A(:, n+1) = X_f(1,:)';
%         X_a_N(:, n+1) = X_f(2,:)';
%         X_a_C(:, n+1) = X_f(3,:)';
%         X_a_N_min(:, n+1) = X_f(4,:)';
%         X_a_kA(:, n+1) = X_f(5,:)';
%     end
% 
%     %Kalman Filter
% %     L = covXF*M'*(inv((M*covXF*M' + diag(measStd))));
% %     X_kf = X_f + L*(x_t - M*X_f);
% %     X_kf_A(:, n+1) = X_kf(1,:)';
% %     X_kf_N(:, n+1) = X_kf(2,:)';
% %     X_kf_C(:, n+1) = X_kf(3,:)';
% 
%     X_f_A(:, n+1) = X_f(1,:)';
%     X_f_N(:, n+1) = X_f(2,:)';
%     X_f_C(:, n+1) = X_f(3,:)';
%     X_f_N_min(:, n+1) = X_f(4,:)';
%     X_f_kA(:, n+1) = X_f(5,:)';
% 
%     x_t_true(:,n+1) = x_t;
% 
% %% Data D-matrix
%     D_A(:,n) = D(1,:)';
%     if(size(M,1) == 2)
%         D_N(:,n) = D(2,:)';
%     end
%     if(size(M,1) == 3)
%         D_N(:,n) = D(2,:)';
%         D_C(:,n) = D(3,:)';
%     end
%     d_true(:,n) = d;
end

% C_content(:, end) = C_content(:, end-1);
% N_content(:, end) = N_content(:, end-1);

%% Net Carbon Fixed
function C = netCarbon (W_s, dCdt)
C = W_s .* dCdt;
end

%% Total gross frond area
function A_tot = totalGross (my, A)
A_tot = my.*A;
end

%% Effect of temperature
function f_temp = EffectTemp (T)
f_temp = zeros(size(T,1), 1);
for n = 1:size(T,1)
if (T(n)>=-1.8) && (T(n) < 10)
    f_temp(n) = 0.08*T(n) + 0.2;
elseif (T(n)>= 10) && (T(n)<= 15)
    f_temp(n) = 1;
elseif (T(n)>15) && (T(n)<=19)
    f_temp(n) = 19/4 - T(n)/4;
elseif (T(n)>19)
    f_temp(n) = 0;
end
end
end

%% Gross photosynthesis
function P = grossPhotosynthesis ( alpha, I_sat, P_1, T_AP, T_R1, T_APL,T_APH, T, I)
T_PL = 271;
T_PH = 296;
P_max = (P_1.*exp(T_AP/T_R1-T_AP./T)) ./ (1+exp(T_APL./T - T_APL/T_PL) + exp(T_APH/T_PH - T_APH./T));
B_vec = zeros(size(T,1), 1);
B_0 = 1*10^(-9);
for i = 1:size(T,1)
    fun = @(B)  -((alpha(i)*I_sat(i)) / (log(1+alpha(i)/B)) * (alpha(i)/(alpha(i)+B)) * ((B/(alpha(i)+B))^(B/alpha(i))) - P_max(i));
    B_new = fminsearch(fun, B_0);
    B_vec(i) = B_new*100;
end

P_S = (alpha.*I_sat)./(log(1+alpha./B_vec));
P = P_S .* (1-exp(-(alpha.*I)./P_S)) .* exp(- (B_vec.*I)./P_S);
end

%% Specific Growth Rate
function my = specificGrowthRate (f_area, f_photo, f_temp, N_min, C_min, N, C)
my = zeros(size(N,1), 1);
N_min_s = 0;
C_min_s = 0;
for n = 1:size(N,1)
if ((1-N_min(n)/N(n)) <= (1-C_min(n)/C(n)))
    my(n) = f_area(n) * f_photo * f_temp(n) * (1-N_min(n)/N(n));
    N_min_s = N_min_s +1;
else
    my(n) = f_area(n) * f_photo * f_temp(n) * (1-C_min(n)/C(n));
    C_min_s = C_min_s +1;
end
end
disp('N_min') 
N_min_s
disp('C_min')
C_min_s
end

%% Model
function [A_dot, N_dot, C_dot, W_s, my, C_total, N_total, C_frac, N_frac, ny] = kelp(N_struct, C_struct, k_A, k_N, k_C, k_dw, N_min, C_min, m_1, m_2, A_O, epsilon, K_X, N_max, J_max, U_065, R_1, T_AR, T_R1, gamma, alpha, I_sat, P_1, T_AP, T_APL,T_APH, U, NO3, T, I, N, C, A)

% Effect of size on growth rate
f_area = m_1.*exp(-(A./A_O).^2) + m_2;

%Effect of temperature on growth rate
f_temp = EffectTemp(T);

% Seasonal influence on growth rate
%f_photo = a_1*(1+sign(lambda)*abs(lambda)^0.5) + a_2;
f_photo = 2;

% Frond erosion
ny = (10.^(-6).*exp(epsilon.*A)) ./ (1+10.^(-6).*(exp(epsilon.*A)-1));

% Nitrate uptake rate
J = J_max.* (NO3./(K_X + NO3)) .* ((N_max-N)./(N_max-N_min)) .* (1-exp(-U./U_065));

% Gross photosynthesis
P = grossPhotosynthesis(alpha, I_sat, P_1, T_AP, T_R1, T_APL,T_APH, T+274.15, I);

% Temperature dependent respiration
R = R_1 .* exp((T_AR/T_R1)- T_AR./(T+274.15));

% Carbon exudation
E = 1 - exp(gamma.*(C_min-C));

% Specific growth rate
my = specificGrowthRate(f_area, f_photo, f_temp, N_min, C_min, N, C);

% Amount of frond area lost
A_lost = A.*(C_min - C) ./ C_struct;

% Structural weight
W_s = k_A.*A;

% Total dry weight
W_d = k_A .* (1+ k_N.*(N-N_min) + N_min + k_C.*(C-C_min) + C_min).*A;

% Total wet weight
W_w = k_A .* (k_dw.^(-1) + k_N.*(N-N_min) + N_min + k_C.*(C-C_min) + C_min).*A;

% Total carbon content
C_total = (C + C_struct) .* W_s;

% Total nitrogen content
N_total = (N + N_struct) .* W_s;

% Carbon content (fraction of dry weight)
C_frac = C_total ./ W_d;

% Nitrogen content (fraction of dry weight)
N_frac = N_total ./W_d;

% Rate of change of frond area
A_dot = (my-ny).*A;

% Rate of change in nitrogen reserves
N_dot = k_A.^(-1).*J-my.*(N+N_struct);

% Rate of change in carbon reserves
C_dot = k_A.^(-1).*(P.*(1-E)-R) - (C+C_struct).*my;

% state_dot = [(my-ny)*A;
%     k_A^(-1)*J-my*(N+N_struct);
%     k_A^(-1)*(P*(1-E-R)) - (C+C_struct)*my];
end