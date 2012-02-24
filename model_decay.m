% Model bitly 'decay' data
%
% Author: Sean Anderson
% Date: 22/02/12

clear all;

% load data
load train;

% number of experimental trials
M = length(Y);

% sample time
dt = 60;

% model order
n = 2;

% initialise variables
T = nan(M,n);   % time constants
x0 = nan(n,M);  % initial conditions
rms = nan(M,1); % root mean square prediction error

% initial time constants
T0 = [100 1000];

% model each trial separately
for i = 1:M
    
    % data
    y = Y{i};
    
    % number of samples
    N = length(y);
    
    % time vector
    t = [0:N-1]'*dt;
    
    % fit time constants assuming exponential model
    fig = 101;
    [T(i,:), x0(:,i), rms(i), e, phi] = fitSumExp(t, y, T0, fig);
    
end



