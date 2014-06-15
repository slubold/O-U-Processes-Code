clc; clf; clear all; close all;

x = 0:pi/10:pi;
y = 0:pi/10:pi;
t = 0:0.01:10;
alpha = 0.5;
CC = 3;

m = numel(y);
n = numel(x);
dim = [m n];

u0 = zeros(m,n);

W = pi; %dimension of rectangle in x-direction
L = pi; %dimension of rectangle in y-direction
k = 0.2; %diffusion constant

for i = 1:n
        u0(1:m,i) = max(5*sin(x(i))*sin(y(1:m))-2,0);
end

figure(1)
[X,Y] = meshgrid(x,y);
mesh(X,Y,u0);
%imagesc(u0);

u0 = reshape(u0, numel(u0),1);

[T, M] = ode45(@(t,u) logisticGrowth_OU(t,u,dim,W,L,k,alpha,CC), t, u0);

for g = 1:length(t)
    figure(2)
    U = reshape(M(g,:),m,n);
    mesh(X,Y,U);
    axis([0,W,0,L,0,3]);
    %imagesc(U);
    %caxis([0,2]);
end

pause(1.5);

close all;