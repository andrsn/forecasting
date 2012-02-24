function [T, x0, erms, e, phi] = fitSumExp(t, y, T0, fig)

% function [T, x0, erms, e, phi] = fitSumExp(t, y, T0, fig)
%
% Estimate parameters for sum of exponential model using separable least
% squares.
%
% t     = vector of times
% y     = data to be fitted
% T0    = initial values for time constants
% fig   = do plot at end (in figure fig)
%
% T      = time constant estimates
% x0     = initial conditions of exponential decay
% erms   = RMS error
% e      = vector of fit errors
% phi    = exponential components (yhat = phi*x0)

% estimate sum of exponential model parameters
w0 = sqrt(1./T0);   % search over square root of parameters
options = optimset('GradObj', 'off', 'Diagnostics', 'off', ...
    'Display', 'off', 'TolFun', eps, 'TolX', eps, 'MaxIter', 1000);
w = fminunc(@expCostFun, w0, options);

    % sum of exponentials cost function
    function [sse] = expCostFun(w)
        
        % parameters constrained to +ve time constants
        lam = w.^2;
        
        % exponential components
        phi = exp(-t*lam);
        
        % prediction error
        e = y-(phi*(phi\y));
        
        % sum of squared error
        sse = sum(e.^2);
        
    end % end cost function

% optimal parameters
lam = w.^2;

% time constants
T = 1./lam;

% exponential components
phi = exp(-t*lam);

% initial conditions
x0 = phi\y;

% prediction error
e = y-(phi*x0);

% root mean squared error
erms = sqrt(sum(e.^2));

% sort time constants in ascending order if necessary
[T, isrt] = sort(T);
x0 = x0(isrt, :);
phi = phi(:, isrt);

% optional figure
if nargin == 4
    figure(fig); clf;
    plot(t, y); hold on;
    plot(t, phi*x0, 'r'); hold off
end

end % end main function




% the end





