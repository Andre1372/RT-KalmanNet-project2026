clear
clc
close all

%% nominal model (CV Model)
dt = 1.0;            % Time step
A = [1 dt;
     0 1];
C = [1 0];
Q = 0.01 * [dt^4/4 dt^3/2;
         dt^3/2 dt^2];
B = [sqrtm(Q) zeros(2,1)];
R = 1;
D = [0 0 sqrtm(R)];
n = size(A,1);
p = size(C,1);

% Inilization for the data generation
V0 = eye(n);
x0 = [0; 1];

%% Parameter
M = 100;  % number of MC cases
T = 2000; % Time interval
c_type = 'constant';    % 'constant' | 'sinusoid' | 'linear'
c_center = 0.1;      %  value        mean value   start value
c_k      = 0.0001;    %  -            amplitude    slope
c_omega  = 0.005;     %  -            frequency    -
c_phi    = 0;         %  -            phase        -

%%%%%%%%%%%%%%%%%%%%%%%%%% data generation %%%%%%%%%%%%%%%%%%%%%%%
% Save nominal data
yn= zeros(p,T,M);
xn= zeros(n,T+1,M);
% Save LFM data
yw= zeros(p,round(T/2),M);
xw= zeros(n,round(T/2)+1,M);


% Nominal data
for k = 1:M
    xn(:,1,k) = x0 + sqrtm(V0) * randn(n,1);   % initial state
    for t = 1:T
        % State update
        xn(:,t+1,k) = A*xn(:,t,k) + mvnrnd(zeros(n,1), Q)';
        % Observation
        yn(:,t,k) = C*xn(:,t,k) + mvnrnd(zeros(p,1), R)';
    end
end

% LFM data
fmt_simple = @(v) regexprep(sprintf('%.4g',v), '\.', '');
if strcmp(c_type, 'constant')
    c = c_center * ones(T,1);
    fname = sprintf('data_%s_%s.mat', c_type, fmt_simple(c_center));
elseif strcmp(c_type, 'sinusoid')
    c = generate_constrained_sinusoid(c_center, c_k, c_omega, c_phi, T);
    fname = sprintf('data_%s_%s_%s_%s_%s.mat', c_type, fmt_simple(c_center), fmt_simple(c_k), fmt_simple(c_omega), fmt_simple(c_phi));
elseif strcmp(c_type, 'linear')
    c = generate_constrained_line(c_center, c_k, T);
    fname = sprintf('data_%s_%s_%s_%s_%s.mat', c_type, fmt_simple(c_center), fmt_simple(c_k));
else
    error('Unknown c_type: %s', c_type);
end

[GR, VR, PR]=robust_filter(A,B,C,D,c,T,V0,x0,yn(:,:,1));
for k = 1:M
    [yw(:,:,k),xw(:,:,k)]=LFM_data(A,B,C,D,GR,PR,VR,x0);
end

% save data
% collapse any accidental multiple underscores
fname = regexprep(fname, '_+', '_');
save(fname, 'A','B','C','D','Q','R','dt','V0','x0', ...
    'M','T','c','c_type','c_center','c_k','c_omega','c_phi', ...
    'yn','xn','yw','xw','GR','VR','PR');

% plot one random trajectory
k_random = randi(M);
figure;

subplot(3,1,1);
plot(xw(1,:,k_random), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Position');
hold on;
plot(xw(2,:,k_random), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Velocity');
xlabel('Time');
ylabel('State');
legend;
grid on;
title(sprintf('State variables (Case %d)', k_random));

subplot(3,1,2);
plot(yw(1,:,k_random), 'g-', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Observation');
grid on;
title('Observation');

subplot(3,1,3);
plot(c(1:round(T/2)), 'm-', 'LineWidth', 1.5);
xlabel('Time');
ylabel('c (true)');
grid on;
title('True c value used for this data (first T/2 instants)');
