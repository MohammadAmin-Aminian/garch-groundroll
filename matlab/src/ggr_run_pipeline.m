function res = ggr_run_pipeline(data, dt, dx, cfg, varargin)
%GGR_RUN_PIPELINE GARCH signal recovery after ground-roll attenuation.
%
%   res = GGR_RUN_PIPELINE(data, dt, dx, cfg) runs the workflow of
%   Aminian & Riahi (2023):
%     1. conventional filter          -> filtered, noise = data - filtered
%     2. GARCH(p,q) per noise trace   -> conditional variance and CSD
%     3. Gabor features + k-means     -> cluster labels of the CSD image
%     4. low-CSD clusters             -> mask (reflection energy in noise)
%     5. recovered = noise.*mask, corrected = filtered + recovered
%
%   CFG is a struct (see GGR_DEFAULT_CONFIG / GGR_LOAD_CONFIG); missing
%   fields take default values. Name-value options:
%     'Filtered' - use this filtered gather instead of cfg.filter
%     'FkMask'   - reject mask for cfg.filter.type = 'fk_mask'
%     'Verbose'  - print progress (default true)
%
%   RES fields: data, filtered, noise, variance, csd, loglik, labels, mask,
%   kept, recovered, corrected, correctedNoise, snr, stats, cfg.

ip = inputParser;
addParameter(ip, 'Filtered', []);
addParameter(ip, 'FkMask', []);
addParameter(ip, 'Verbose', true);
parse(ip, varargin{:});
o = ip.Results;
if nargin < 4, cfg = struct(); end
cfg = ggr_merge_config(ggr_default_config(), cfg);
data = double(data);

if isempty(o.Filtered)
    say(o.Verbose, '1/4 conventional filter: %s\n', cfg.filter.type);
    filtered = ggr_apply_filter(data, dt, dx, cfg.filter, o.FkMask);
else
    filtered = double(o.Filtered);
end
noise = data - filtered;

p = cfg.garch.p; q = cfg.garch.q;
say(o.Verbose, '2/4 GARCH(%d,%d) on %d traces\n', p, q, size(noise, 2));
g = ggr_garch_traces(noise, p, q, o.Verbose);
csd = g.std;

say(o.Verbose, '3/4 Gabor features + k-means\n');
fc = cfg.features;
X = ggr_build_features(csd, fc.wavelengths, fc.orientations, fc.smoothing, ...
    fc.coordinates, fc.scaling);
cc = cfg.clustering;
labels = reshape(ggr_kmeans(X, cc.n_clusters, cc.n_init, 100, cc.seed), size(csd));

switch cc.selection
    case 'n_keep',    param = cc.n_keep;
    case 'threshold', param = cc.threshold;
    case 'manual',    param = cc.clusters;
    otherwise,        param = [];
end
[mask, kept] = ggr_select_clusters(labels, csd, cc.selection, param);
say(o.Verbose, '4/4 signal recovery, clusters kept: %s\n', mat2str(kept'));

recovered = noise .* mask;
corrected = filtered + recovered;
correctedNoise = noise - recovered;

res = struct('data', data, 'filtered', filtered, 'noise', noise, ...
    'variance', g.variance, 'csd', csd, 'loglik', g.loglik, 'labels', labels, ...
    'mask', mask, 'kept', kept, 'recovered', recovered, 'corrected', corrected, ...
    'correctedNoise', correctedNoise, ...
    'snr', ggr_snr_report(filtered, noise, corrected, correctedNoise), ...
    'stats', ggr_cluster_stats(labels, csd), 'cfg', cfg);
end

function say(v, varargin)
if v, fprintf(varargin{:}); end
end
