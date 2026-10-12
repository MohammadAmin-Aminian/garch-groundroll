function fit = ggr_garch_fit(y, p, q, nStarts)
%GGR_GARCH_FIT Maximum-likelihood fit of a zero-mean GARCH(p,q) model.
%
%   fit = GGR_GARCH_FIT(y, p, q) fits
%       y(t) = sigma(t) z(t),  z ~ N(0,1)
%       sigma2(t) = omega + sum_{i=1..q} alpha(i) y(t-i)^2
%                         + sum_{j=1..p} beta(j) sigma2(t-j)
%   P is the number of GARCH (lagged variance) terms and Q the number of
%   ARCH terms, as in the Econometrics Toolbox GARCH(P,Q). No toolbox is
%   needed: the likelihood is maximised with FMINSEARCH under the
%   re-parameterisation
%       omega = exp(th0),  [alpha beta] = c*exp(th)/(1+sum(exp(th))), c<1
%   which enforces omega>0, alpha,beta>=0 and sum(alpha)+sum(beta)<1.
%   The series is scaled to unit RMS before fitting and the estimates are
%   mapped back exactly. NSTARTS (1-3, default 3) starting points are tried.
%
%   Output struct fields: p, q, omega, alpha, beta, loglik, variance, std,
%   nobs, aic, bic, converged.
%
%   See also GGR_GARCH_TRACES, GGR_GARCH_VARIANCE.

if nargin < 2 || isempty(p), p = 1; end
if nargin < 3 || isempty(q), q = 1; end
if nargin < 4 || isempty(nStarts), nStarts = 3; end
if q < 1 || p < 0
    error('ggr:garch', 'Require q >= 1 (ARCH terms) and p >= 0 (GARCH terms).');
end

y = double(y(:));
n = numel(y);
scale = sqrt(mean(y .^ 2));
fit = struct('p', p, 'q', q, 'omega', 0, 'alpha', zeros(q, 1), 'beta', zeros(p, 1), ...
    'loglik', 0, 'variance', zeros(n, 1), 'std', zeros(n, 1), 'nobs', n, ...
    'aic', NaN, 'bic', NaN, 'converged', false);
if n < 3 * (p + q + 1) || scale == 0 || ~isfinite(scale)
    return
end

z = y / scale;
nll = @(th) negloglik(th, z, p, q);
starts = [0.10 0.80; 0.50 0.45; 0.90 0.05];
opts = optimset('Display', 'off', 'TolX', 1e-8, 'TolFun', 1e-9, ...
    'MaxIter', 4000 * (p + q + 1), 'MaxFunEvals', 8000 * (p + q + 1));

best = Inf;
thBest = [];
flagBest = 0;
for s = 1:max(1, min(nStarts, size(starts, 1)))
    a = starts(s, 1);
    b = starts(s, 2);
    if p == 0
        a = a + b;
        b = 0;
    end
    th0 = startValues(p, q, a, b);
    [th, fval, flag] = fminsearch(nll, th0, opts);
    % restart once from the optimum (standard remedy for Nelder-Mead stalls)
    [th2, fval2, flag2] = fminsearch(nll, th, opts);
    if fval2 <= fval
        th = th2; fval = fval2; flag = flag2;
    end
    if isfinite(fval) && fval < best
        best = fval;
        thBest = th;
        flagBest = flag;
    end
end

[omega, alpha, beta] = unpack(thBest, p, q);
s2 = ggr_garch_variance(z, omega, alpha, beta, 1);
llz = loglik(z, s2);

fit.omega = omega * scale ^ 2;
fit.alpha = alpha;
fit.beta = beta;
fit.loglik = llz - n * log(scale);
fit.variance = s2 * scale ^ 2;
fit.std = sqrt(fit.variance);
k = 1 + p + q;
fit.aic = -2 * fit.loglik + 2 * k;
fit.bic = -2 * fit.loglik + log(n) * k;
fit.converged = flagBest > 0;
end

% -------------------------------------------------------------------------
function v = negloglik(th, z, p, q)
[omega, alpha, beta] = unpack(th, p, q);
s2 = ggr_garch_variance(z, omega, alpha, beta, 1);
v = -loglik(z, s2);
if ~isfinite(v), v = 1e300; end
end

function ll = loglik(y, s2)
s2 = max(s2, realmin);
ll = -0.5 * sum(log(2 * pi) + log(s2) + y .^ 2 ./ s2);
end

function [omega, alpha, beta] = unpack(th, p, q)
c = 0.9999;
th = th(:);
omega = exp(min(max(th(1), -50), 50));
e = exp(min(max(th(2:end), -50), 50));
w = c * e / (1 + sum(e));
alpha = w(1:q);
beta = w(q+1:q+p);
end

function th = startValues(p, q, aSum, bSum)
c = 0.9999;
alpha = repmat(aSum / q, q, 1);
if p > 0
    beta = repmat(bSum / p, p, 1);
else
    beta = zeros(0, 1);
end
omega = 1 - sum(alpha) - sum(beta);
w = max([alpha; beta] / c, 1e-8);
rest = max(1 - sum(w), 1e-8);
th = [log(omega); log(w / rest)];
end
