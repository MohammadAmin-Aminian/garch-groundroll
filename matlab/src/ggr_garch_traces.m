function out = ggr_garch_traces(noise, p, q, verbose)
%GGR_GARCH_TRACES Fit an independent GARCH(p,q) model to every trace.
%
%   out = GGR_GARCH_TRACES(noise, p, q) returns a struct with fields
%   variance (nt x nx), std (nt x nx), loglik (1 x nx), omega (1 x nx),
%   alpha (q x nx), beta (p x nx), converged (1 x nx).
%
%   See also GGR_GARCH_FIT.

if nargin < 2 || isempty(p), p = 1; end
if nargin < 3 || isempty(q), q = 1; end
if nargin < 4, verbose = false; end
[nt, nx] = size(noise);
out.variance = zeros(nt, nx);
out.loglik = zeros(1, nx);
out.omega = zeros(1, nx);
out.alpha = zeros(q, nx);
out.beta = zeros(p, nx);
out.converged = false(1, nx);
for j = 1:nx
    f = ggr_garch_fit(noise(:, j), p, q);
    out.variance(:, j) = f.variance;
    out.loglik(j) = f.loglik;
    out.omega(j) = f.omega;
    out.alpha(:, j) = f.alpha;
    out.beta(:, j) = f.beta;
    out.converged(j) = f.converged;
    if verbose && mod(j, 50) == 0
        fprintf('    GARCH(%d,%d): %d / %d traces\n', p, q, j, nx);
    end
end
out.std = sqrt(out.variance);
out.p = p;
out.q = q;
end
