function [mask, kept] = ggr_select_clusters(labels, csd, method, param)
%GGR_SELECT_CLUSTERS Choose the clusters that contain leaked reflections.
%
%   [mask, kept] = GGR_SELECT_CLUSTERS(labels, csd, method, param)
%   Ground roll has a large conditional standard deviation (CSD), reflections
%   a small one, so clusters are ranked by mean CSD. METHOD:
%     'gap'       (default) split the sorted cluster means at the largest
%                 jump; clusters below the jump are signal.
%     'n_keep'    keep the PARAM lowest-CSD clusters.
%     'threshold' keep clusters with mean CSD < PARAM * max(mean CSD).
%     'manual'    keep the cluster ids listed in PARAM.
%   MASK is 1 where the removed noise is reclassified as signal.

if nargin < 3 || isempty(method), method = 'gap'; end
s = ggr_cluster_stats(labels, csd);
[m, order] = sort(s.meanCsd);
switch lower(method)
    case 'gap'
        if numel(m) > 1
            [~, cut] = max(diff(m));
        else
            cut = 0;
        end
        kept = sort(s.ids(order(1:cut)));
    case 'n_keep'
        if isempty(param) || param < 1 || param >= numel(s.ids)
            error('ggr:select', 'n_keep must be between 1 and n_clusters-1');
        end
        kept = sort(s.ids(order(1:param)));
    case 'threshold'
        kept = sort(s.ids(s.meanCsd < param * max(s.meanCsd)));
    case 'manual'
        kept = sort(param(:));
    otherwise
        error('ggr:select', 'Unknown method %s', method);
end
mask = double(ismember(labels, kept));
end
