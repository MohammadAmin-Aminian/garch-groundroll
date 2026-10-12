function sigma2 = ggr_garch_variance(y, omega, alpha, beta, backcast)
%GGR_GARCH_VARIANCE Conditional variance of a zero-mean GARCH(p,q) series.
%
%   sigma2 = GGR_GARCH_VARIANCE(y, omega, alpha, beta) returns
%       sigma2(t) = omega + sum_i alpha(i)*y(t-i)^2 + sum_j beta(j)*sigma2(t-j)
%   with length(alpha) = q (ARCH terms) and length(beta) = p (GARCH terms).
%   Pre-sample values of y^2 and sigma^2 equal BACKCAST (default mean(y.^2)).
%   The recursion is evaluated with FILTER (vectorised).
%
%   See also GGR_GARCH_FIT.

y = y(:);
alpha = alpha(:);
beta = beta(:);
q = numel(alpha);
p = numel(beta);
n = numel(y);
y2 = y .^ 2;
if nargin < 5 || isempty(backcast)
    backcast = mean(y2);
end

% x(t) = omega + sum_i alpha(i) * y(t-i)^2   (pre-sample y^2 = backcast)
y2e = [repmat(backcast, q, 1); y2];
x = repmat(omega, n, 1);
for i = 1:q
    x = x + alpha(i) * y2e(q-i+1:q-i+n);
end
if p == 0
    sigma2 = x;
    return
end
% sigma2(t) = x(t) + sum_j beta(j) * sigma2(t-j)   (pre-sample sigma2 = backcast)
zi = backcast * flipud(cumsum(flipud(beta)));     % direct-form-II-transposed state
sigma2 = filter(1, [1; -beta], x, zi);
end
