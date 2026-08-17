function [y,x]=LFM_data(A,B,C,D,G,P,V,x0)

% the function returns 
% x: least favroable state
% y: least favroable measurement 

n=size(A,1);
p=size(C,1);
T=size(G,n+p);
%% init 
Om_inv=zeros(n,n);

%% backward recursion
for i=T:-1:1
    Phi(:,:,i)=P(:,:,i+1)^(-1)-V(:,:,i+1)^(-1);
    K(:,:,i)=(eye(n+p)-(B-G(:,:,i)*D)'*(Om_inv+Phi(:,:,i))*(B-G(:,:,i)*D))^-1;
    H(:,:,i)=K(:,:,i)*(B-G(:,:,i)*D)'*(Om_inv+Phi(:,:,i))*(A-G(:,:,i)*C);
    L(:,:,i)=chol(K(:,:,i))';
    Ae(:,:,i)=[A B*H(:,:,i); zeros(n,n) A-G(:,:,i)*C+(B-G(:,:,i)*D)*H(:,:,i)];
    Be(:,:,i)=[B; B-G(:,:,i)*D]*L(:,:,i);
    Ce(:,:,i)=[C D*H(:,:,i)];
    De(:,:,i)=D*L(:,:,i);
    Om_inv=(A-G(:,:,i)*C)'*(Om_inv+Phi(:,:,i))*(A-G(:,:,i)*C)+H(:,:,i)'*K(:,:,i)^-1*H(:,:,i);
end

%% forward recursion
T  = round(T/2); % N
ww = randn(n+p,T);  
y  = zeros(p,T); 
x  = zeros(n,T+1); 
x(:,1) = x0 + sqrtm(V(:,:,1)) * randn(n,1);   % initial state 
xm(:,1) = [x(:,1) ; zeros(n,1)]; %the initial of state \bar x_0.
for t=1:T
     xm(:,t+1) = Ae(:,:,t) * xm(:,t) + Be(:,:,t)*ww(:,t); %x_t+1 (0:N+1)
     y(:,t) = Ce(:,:,t) * xm(:,t) + De(:,:,t)*ww(:,t); %y_t (0:N)
     x(:,t+1) = xm(1:n,t+1);
end
end





