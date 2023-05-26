%% States
% x = [A; N; C]
% environmental variables = [T; I; NO3; C]
% f = [dA; dN; dC] ?
% 

%% Initialize step

%% Prediction step

%% Update step


%% From PPT
function x = EnKf ()
    for sample = 1:NumberIterations
        t = (sample-1)*dt;

        % Integrate "true" process
        % x_t: "true" values
        x_t = x_t + dt*f_j(t,x_t);

        % Calculate forecast value
        % X_f: ensemble state
        for j = 1:Nsample
            X_f (:,j) = X_a(:,j) + dt*f_j(t, X_a(:,j)); % Step all ensemble memebers
        end
    end

    % Analysis step:
    d = M*x_t; % Measurement without error

    %Set up measurement ensemble D with errors added: 
    for j = 1:N
        D(:,j) = d + measStd .* randn(size(measStd));
    end

    %Compute intermediate matrices: 
    MX = M*X_f;
    MA = MX - (1/N)*(MX*ones(Nsample,1))*ones(1,Nsample);
    P = (1/(N-1))*MA*MA' + C_ee;
    A = X_f - (1/Nsample)*X_f*ones(Nsample,1)*ones(1,Nsample);

    %Compute analysis: 
    X_a = X_f + (1/(1+Nsample))*A*MA'*inv(P)*(D-MX);


end

%%