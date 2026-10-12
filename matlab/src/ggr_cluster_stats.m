function s = ggr_cluster_stats(labels, csd)
%GGR_CLUSTER_STATS Mean CSD and pixel fraction of each cluster.
ids = unique(labels(:));
s.ids = ids;
s.meanCsd = zeros(size(ids));
s.fraction = zeros(size(ids));
for i = 1:numel(ids)
    m = labels == ids(i);
    s.meanCsd(i) = mean(csd(m));
    s.fraction(i) = mean(m(:));
end
end
