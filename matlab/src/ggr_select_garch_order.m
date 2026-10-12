function sel = ggr_select_garch_order(noise, orders, criterion)
%GGR_SELECT_GARCH_ORDER Compare GARCH(p,q) models trace by trace.
%
%   sel = GGR_SELECT_GARCH_ORDER(noise) fits the nine models p,q = 1..3.
%   ORDERS is an m x 2 matrix of [p q] rows. CRITERION is 'bic' (default),
%   'aic' or 'loglik'. 'loglik' reproduces the comparison in the paper; as
%   the models are nested it always favours the largest model, so BIC is
%   the default.
%
%   Fields: orders, loglik (m x nx), score (m x nx, lower is better),
%   best (1 x nx index into orders), counts (m x 1), bestOverall ([p q]).

if nargin < 2 || isempty(orders)
    [Q, P] = meshgrid(1:3, 1:3);
    orders = [reshape(P', [], 1), reshape(Q', [], 1)];
end
if nargin < 3 || isempty(criterion), criterion = 'bic'; end
[nt, nx] = size(noise);
m = size(orders, 1);
ll = zeros(m, nx);
score = zeros(m, nx);
for k = 1:m
    p = orders(k, 1);
    q = orders(k, 2);
    r = ggr_garch_traces(noise, p, q);
    ll(k, :) = r.loglik;
    npar = 1 + p + q;
    switch lower(criterion)
        case 'bic',    score(k, :) = -2 * ll(k, :) + log(nt) * npar;
        case 'aic',    score(k, :) = -2 * ll(k, :) + 2 * npar;
        case 'loglik', score(k, :) = -ll(k, :);
        otherwise, error('ggr:criterion', 'criterion must be bic, aic or loglik');
    end
end
[~, best] = min(score, [], 1);
counts = accumarray(best(:), 1, [m 1]);
[~, iBest] = max(counts);
sel = struct('orders', orders, 'loglik', ll, 'score', score, 'criterion', criterion, ...
    'best', best, 'counts', counts, 'bestOverall', orders(iBest, :));
end
