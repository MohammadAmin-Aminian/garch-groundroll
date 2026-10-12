function run_tests()
%RUN_TESTS Unit tests of the MATLAB toolbox (no test framework needed).
%   >> cd matlab/tests; run_tests
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup.m'));
tests = {@testGarchRecovery, @testGarchVarianceRecursion, @testOrmsby, ...
    @testFkMaskIdentity, @testFkFanKeepsFastEvents, @testKmeans, @testSelectClusters, ...
    @testSnr, @testLocalCorrelation, @testSyntheticGeometry, @testPipelineSynthetic};
nFail = 0;
for i = 1:numel(tests)
    name = func2str(tests{i});
    try
        tests{i}();
        fprintf('  PASS  %s\n', name);
    catch err
        nFail = nFail + 1;
        fprintf('  FAIL  %s\n        %s\n', name, err.message);
    end
end
fprintf('\n%d of %d tests passed\n', numel(tests) - nFail, numel(tests));
if nFail > 0, error('ggr:tests', '%d test(s) failed', nFail); end
end

function y = simulateGarch(n, omega, a, b, seed)
rs = RandStream('mt19937ar', 'Seed', seed);
y = zeros(n, 1);
s2 = omega / (1 - a - b);
for t = 1:n
    y(t) = sqrt(s2) * randn(rs);
    s2 = omega + a * y(t) ^ 2 + b * s2;
end
end

function testGarchRecovery()
y = 7 * simulateGarch(5000, 0.2, 0.12, 0.80, 1);
f = ggr_garch_fit(y, 1, 1);
assert(abs(f.alpha - 0.12) < 0.05, 'alpha not recovered');
assert(abs(f.beta - 0.80) < 0.08, 'beta not recovered');
assert(abs(f.omega / 49 - 0.2) < 0.15, 'omega not recovered');
assert(f.alpha + f.beta < 1, 'stationarity violated');
end

function testGarchVarianceRecursion()
y = [1; -2; 0.5; 3];
s2 = ggr_garch_variance(y, 0.1, 0.2, 0.7, 1);
ref = zeros(4, 1); prevS = 1; prevY = 1;
for t = 1:4
    ref(t) = 0.1 + 0.2 * prevY + 0.7 * prevS;
    prevS = ref(t); prevY = y(t) ^ 2;
end
assert(max(abs(s2 - ref)) < 1e-12, 'GARCH(1,1) recursion wrong');
end

function testOrmsby()
dt = 0.002; t = (0:999)' * dt;
lo = sin(2 * pi * 5 * t); hi = sin(2 * pi * 60 * t);
out = ggr_ormsby(lo + hi, dt, [0 0 15 25], 'bandstop');
c = 100:900;
assert(sqrt(mean((out(c) - hi(c)) .^ 2)) < 0.05, 'band-stop did not remove 5 Hz / keep 60 Hz');
bp = ggr_ormsby(lo + hi, dt, [0 0 15 25], 'bandpass');
assert(max(abs(bp + out - (lo + hi))) < 1e-10, 'bandpass + bandstop must sum to input');
end

function testFkMaskIdentity()
d = randn(64, 32);
assert(max(max(abs(ggr_fk_mask(d, zeros(64, 32)) - d))) < 1e-12);
assert(max(max(abs(ggr_fk_mask(d, ones(64, 32))))) < 1e-10);
end

function testFkFanKeepsFastEvents()
dt = 0.004; dx = 10; nt = 256; nx = 64;
t = (0:nt-1)' * dt; x = (0:nx-1) * dx;
fast = exp(-((t - 0.4 - x / 5000) / 0.02) .^ 2);
slow = exp(-((t - 0.1 - x / 300) / 0.05) .^ 2);
out = ggr_fk_fan(fast + slow, dt, dx, 800, 1200);
assert(norm(out - fast, 'fro') / norm(slow, 'fro') < 0.35, 'fan filter failed');
end

function testKmeans()
rs = RandStream('mt19937ar', 'Seed', 3);
X = [randn(rs, 300, 2); randn(rs, 100, 2) + 8];
lab = ggr_kmeans(X, 2, 3, 100, 0);
assert(all(lab(1:300) == 1) && all(lab(301:400) == 2), 'k-means clusters wrong');
end

function testSelectClusters()
labels = [1 1 2 2; 3 3 4 4];
csd = [0.1 0.1 1.0 1.0; 0.2 0.2 0.9 0.9];
[mask, kept] = ggr_select_clusters(labels, csd, 'gap');
assert(isequal(kept(:)', [1 3]), 'gap selection wrong');
assert(isequal(mask, [1 1 0 0; 1 1 0 0]));
[~, k2] = ggr_select_clusters(labels, csd, 'n_keep', 1);
assert(isequal(k2, 1));
end

function testSnr()
assert(abs(ggr_snr_db(ones(10, 1), 0.1 * ones(10, 1)) - 20) < 1e-12);
end

function testLocalCorrelation()
a = randn(50, 40);
c = ggr_local_correlation(a, 3 * a + 1, 9);
assert(min(c(:)) > 0.999, 'identical images must correlate to 1');
end

function testSyntheticGeometry()
g = ggr_make_synthetic();
assert(isequal(size(g.data), [1501 100]));
[~, i] = max(g.signal(1:350, 51));          % first reflection, offset +20 m
assert(abs((i - 1) * g.dt - sqrt(0.5 ^ 2 + (20 / 1800) ^ 2)) < 0.004, 'first reflection misplaced');
assert(max(abs(g.data(:) - g.signal(:) - g.groundRoll(:))) < 1e-12);
end

function testPipelineSynthetic()
g = ggr_make_synthetic();
cfg = ggr_load_config('synthetic');
res = ggr_run_pipeline(g.data, g.dt, g.dx, cfg, 'Verbose', false);
before = ggr_true_snr_db(res.filtered, g.signal);
after = ggr_true_snr_db(res.corrected, g.signal);
assert(after > before + 1, sprintf('recovery should improve true SNR (%.2f -> %.2f dB)', before, after));
assert(max(abs(res.corrected(:) - res.filtered(:) - res.recovered(:))) < 1e-12);
end
