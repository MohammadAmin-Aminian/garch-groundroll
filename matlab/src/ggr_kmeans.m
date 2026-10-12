function labels = ggr_kmeans(X, k, nInit, maxIter, seed)
%GGR_KMEANS k-means++ clustering of the rows of X (no toolbox required).
%
%   labels = GGR_KMEANS(X, k, nInit, maxIter, seed) returns labels 1..k
%   (column vector), keeping the best of NINIT runs (default 10). Clusters
%   are renumbered by decreasing size so that cluster 1 is the largest.

if nargin < 3 || isempty(nInit), nInit = 10; end
if nargin < 4 || isempty(maxIter), maxIter = 100; end
if nargin < 5 || isempty(seed), seed = 0; end
prev = rng;
rng(seed, 'twister');
cleanup = onCleanup(@() rng(prev));

N = size(X, 1);
x2 = sum(X .^ 2, 2);
bestInertia = Inf;
best = [];
for r = 1:nInit
    % k-means++ seeding
    C = X(randi(N), :);
    dmin = max(x2 - 2 * X * C' + sum(C .^ 2, 2)', 0);
    for j = 2:k
        pr = cumsum(dmin) / sum(dmin);
        idx = find(pr >= rand, 1, 'first');
        if isempty(idx), idx = randi(N); end
        C(j, :) = X(idx, :);
        dmin = min(dmin, max(x2 - 2 * X * C(j, :)' + sum(C(j, :) .^ 2), 0));
    end
    lab = zeros(N, 1);
    for it = 1:maxIter
        D = x2 - 2 * X * C' + sum(C .^ 2, 2)';
        [dmin, newLab] = min(D, [], 2);
        if isequal(newLab, lab), break; end
        lab = newLab;
        for j = 1:k
            members = lab == j;
            if any(members)
                C(j, :) = mean(X(members, :), 1);
            else
                [~, far] = max(dmin);            % re-seed an empty cluster
                C(j, :) = X(far, :);
            end
        end
    end
    inertia = sum(max(dmin, 0));
    if inertia < bestInertia
        bestInertia = inertia;
        best = lab;
    end
end
counts = accumarray(best, 1, [k 1]);
[~, order] = sort(counts, 'descend');
remap = zeros(k, 1);
remap(order) = 1:k;
labels = remap(best);
end
