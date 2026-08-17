function [t] = theta(P,c)

% the function returns 
% t theta at time t

n=size(P,1);
value=1;
t1=0;
e = eig(P);
r = max(abs(e));
t2=(1-10^-5)*(r)^-1;
while abs(value)>=10^-9
    t=0.5*(t1+t2);
    value=trace(inv(eye(n)-t*P)-eye(n)) + log(det(eye(n) -t*P))-c;
    if value>0
        t2=t;
    else
        t1=t;
    end
end
end

