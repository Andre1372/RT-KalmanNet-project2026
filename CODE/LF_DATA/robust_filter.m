function [G, V, P, Y]=robust_filter(A,B,C,D,c,T,V0,x,y)

% the function returns 
% G_t t\in[1,T]
% V_t t\in[1,T+1]
% P_t t\in [1,T+1] P1 is not defined
% y_t t\in[1,T]

Q= B*B';
R= D*D';
n= size(A,1);
m= size(C,1);
V= zeros(n,n,T+1);
P= zeros(n,n,T+1);
G= zeros(n,m,T);
Y= zeros(m,T);

% init
V(:,:,1)=V0;

% iterative part
for i=1:T
    G(:,:,i) = A*V(:,:,i)*C'*inv(C*V(:,:,i)*C'+R);
    Y(:,i) = C * x;
    x = A * x + G(:,:,i) * (y(:,i) - C * x);
    P(:,:,i+1) = (A-G(:,:,i)*C)*V(:,:,i)*(A-G(:,:,i)*C)'...
        +(B-G(:,:,i)*D)*(B-G(:,:,i)*D)';
    % theta_t
    th = theta(P(:,:,i+1),c(i)); %c(i)
    V(:,:,i+1) = inv(inv(P(:,:,i+1))-th*eye(n));
end
