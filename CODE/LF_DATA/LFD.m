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
B = [sqrt(Q) zeros(2,1)];
R = 1;
D = [0 0 sqrt(R)];
n = size(A,1);
p = size(C,1);

% Inilization for the data generation
V0 = eye(n);
x0 = [0; 1];  

%% Parameter
M = 100;  % number of MC cases
T = 2000; % Time interval
cM= 10;   % number of different c

%%%%%%%%%%%%%%%%%%%%%%%%%% data generation %%%%%%%%%%%%%%%%%%%%%%%
% Save all the true c
c = zeros(cM,1);
% Save nominal data 
yn= zeros(p,T,M); 
xn= zeros(n,T+1,M);
% Save LFM data
yw= zeros(p,round(T/2),M,cM);
xw= zeros(n,round(T/2)+1,M,cM);

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
for j = 1:cM
    c(j) = 1;
    [GR, VR, PR]=robust_filter(A,B,C,D,c(j),T,V0,x0,yn(:,:,1));
    for k = 1:M
        [yw(:,:,k,j),xw(:,:,k,j)]=LFM_data(A,B,C,D,GR,PR,VR,x0);
    end
end

save data