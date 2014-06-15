function uprime = logisticGrowth_OU_Diffusion(~,u, dim, W, L, kk, alpha, CC)
m = dim(1);
n = dim(2);
deltaX = W/n;
deltaY = L/m;
k_bar = 2; % Average diffusion rate
beta = 1; % Coefficient on dW
temp = zeros(m, n);
u = reshape(u, m, n);
%% Solves the OU equation and outputs x(t).
%EM  Euler-Maruyama method on linear SDE

% SDE is  dX = lambda*X dt + mu*X dW,   X(0) = Xzero,
%      where lambda = 2, mu = 1 and Xzero = 1.
%
% Discretized Brownian path over [0,1] has dt = 2^(-8).
% Euler-Maruyama uses timestep dt.

randn('state',100)
% T = 1; 
N = 2^8; 
dt = 1/N;   
theta = 1;
K = .015;
sigma = .9;

dW = sqrt(dt)*randn(1,m*n);         % Brownian increments
% W = cumsum(dW);                   % discretized Brownian path 

% Xtrue = Xzero*exp((lambda-0.5*mu^2) * ([dt:dt:T]) + mu.*W); 
% plot([0:dt:T],[Xzero,Xtrue],'m-'), hold on
% hold off

%Xem = zeros(m,n);                 % preallocate for efficiency
tempp = zeros(m*n,1);

% tempp(1) = Xzero + dt*lambda*Xzero + mu*Xzero*dW(1);
tempp(1) = 0;

for j = 2:m*n
    tempp(j) = tempp(j-1)+dt.*K*(theta-tempp(j-1))+sigma*dW(j);
end 
Xem = reshape(tempp,m,n);


%%
temp(1,1) = (kk.*Xem(1,1)).*((u(1,1+1) -2*u(1,1) + u(1,1))/deltaX.^2 + ...
    (u(1+1,1) -2*u(1,1) + u(1,1))/deltaY.^2) + Xem(1,1)*u(1,1)*(CC - u(1,1));
temp(1,n) = (kk.*Xem(j,n)).*((u(1,n) -2*u(1,n) + u(1,n-1))/deltaX.^2 + ...
    (u(1+1,n) -2*u(1,n) + u(1,n))/deltaY.^2) + Xem(1,n)*u(1,n)*(CC - u(1,n));
temp(m,1) = (kk.*Xem(m,1)).*((u(m,1+1) -2*u(m,1) + u(m,1))/deltaX.^2 + ...
    (u(m,1) -2*u(m,1) + u(m-1,1))/deltaY.^2) + Xem(m,1)*u(m,1)*(CC - u(m,1));
temp(m,n) = (kk.*Xem(m,n)).*((u(m,n) -2*u(m,n) + u(m,n-1))/deltaX.^2 + ...
    (u(m,n) -2*u(m,n) + u(m-1,n))/deltaY.^2) + Xem(m,n)*u(m,n)*(CC - u(m,n));

for i = 2:n-1
    temp(1,i) = (kk.*Xem(1,i)).*((u(1,i+1) -2*u(1,i) + u(1,i-1))/deltaX.^2 + ...
        (u(1+1,i) -2*u(1,i) + u(1,i))/deltaY.^2) + Xem(1,i)*u(1,i)*(CC - u(1,i));
    temp(m,i) = (kk.*Xem(m,i))*((u(m,i+1) -2*u(m,i) + u(m,i-1))/deltaX.^2 + ...
        (u(m,i) -2*u(m,i) + u(m-1,i))/deltaY.^2) + Xem(m,i)*u(m,i)*(CC - u(m,i));
end

for j = 2:m-1
    temp(j,1) = (kk.*Xem(j,1)).*((u(j,1+1) -2*u(j,1) + u(j,1))/deltaX.^2 + ...
        (u(j+1,1) -2*u(j,1) + u(j-1,1))/deltaY.^2) + Xem(j,1)*u(j,1)*(CC - u(j,1));
    temp(j,n) = (kk.*Xem(j,n)).*((u(j,n) -2*u(j,n) + u(j,n-1))/deltaX.^2 + ...
        (u(j+1,n) -2*u(j,n) + u(j-1,n))/deltaY.^2) + Xem(j,n)*u(j,n)*(CC - u(j,n));
end

for i = 2:n-1;
    for j = 2:m-1;
        temp(j, i) =  (kk.*Xem(j,i)).*((u(j,i+1) -2*u(j,i) + u(j,i-1))/deltaX.^2 + ...
        (u(j+1,i) -2*u(j,i) + u(j-1,i))/deltaY.^2) + ...
        Xem(j,i).*u(j,i)*(CC - u(j,i));
    end
end

tempp = reshape(temp, m*n, 1);
uprime = tempp;

end