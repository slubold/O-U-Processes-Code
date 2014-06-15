
randn('state',100)
lambda = .5;                        % problem parameters
mu = 1; 
Xzero = 1;
T = 1; 
N = 2^8; 
dt = 1/N;         

dW = sqrt(dt)*randn(1,N);         % Brownian increments
W = cumsum(dW);                   % discretized Brownian path 

Xtrue = Xzero*exp((lambda-0.5*mu^2)*([dt:dt:T])+mu*W); 
plot([0:dt:T],[Xzero,Xtrue],'m-'), hold on