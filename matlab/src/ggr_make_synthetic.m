function g = ggr_make_synthetic(opts)
%GGR_MAKE_SYNTHETIC Split-spread shot gather: reflections + dispersive ground roll.
%
%   g = GGR_MAKE_SYNTHETIC() reproduces the synthetic model of the study:
%   5 hyperbolic reflections (t0 = 0.5 0.9 1.5 1.7 2.0 s, v = 1800 1800 1700
%   1700 1600 m/s, amplitude 3, 20 Hz Ricker), 2 ms sampling, 1501 samples,
%   100 channels at 20 m (offsets -1000..-20, 20..1000 m), and a dispersive
%   ground-roll train (6->14 Hz sweep, 1000->600 m/s, amplitude 3).
%   OPTS (struct) may override any of: nt, dt, nSide, dx, t0, velocities,
%   amplitudes, f0, grT0, grDuration, grVelocity, grFreq, grAmplitude,
%   noiseStd, seed.
%
%   Output fields: data, signal, groundRoll, offsets, dt, dx.

d = struct('nt', 1501, 'dt', 0.002, 'nSide', 50, 'dx', 20, ...
    't0', [0.5 0.9 1.5 1.7 2.0], 'velocities', [1800 1800 1700 1700 1600], ...
    'amplitudes', [3 3 3 3 3], 'f0', 20, 'grT0', 0.4, 'grDuration', 0.5, ...
    'grVelocity', [1000 600], 'grFreq', [6 14], 'grAmplitude', 3, ...
    'noiseStd', 0, 'seed', 0);
if nargin >= 1
    fn = fieldnames(opts);
    for i = 1:numel(fn), d.(fn{i}) = opts.(fn{i}); end
end
offsets = [-(d.nSide:-1:1), 1:d.nSide] * d.dx;
nx = numel(offsets);
t = (0:d.nt-1)' * d.dt;
w = ggr_ricker(d.f0, d.dt);

signal = zeros(d.nt, nx);
for e = 1:numel(d.t0)
    tt = sqrt(d.t0(e) ^ 2 + (offsets / d.velocities(e)) .^ 2);
    for j = 1:nx
        signal(:, j) = placeWavelet(signal(:, j), tt(j), d.dt, w, d.amplitudes(e));
    end
end

gr = zeros(d.nt, nx);
vFast = d.grVelocity(1); vSlow = d.grVelocity(2);
fa = d.grFreq(1); fb = d.grFreq(2);
for j = 1:nx
    x = abs(offsets(j));
    ts = d.grT0 + x / vFast;
    te = d.grT0 + d.grDuration + x / vSlow;
    m = t >= ts & t <= te;
    if ~any(m), continue; end
    tau = (t(m) - ts) / (te - ts);
    phase = 2 * pi * (te - ts) * (fa * tau + 0.5 * (fb - fa) * tau .^ 2);
    gr(m, j) = sin(phase) .* sin(pi * tau) .^ 2;
end
gr = gr * d.grAmplitude;

data = signal + gr;
if d.noiseStd > 0
    rs = RandStream('mt19937ar', 'Seed', d.seed);
    data = data + d.noiseStd * randn(rs, size(data));
end
g = struct('data', data, 'signal', signal, 'groundRoll', gr, 'offsets', offsets, ...
    'dt', d.dt, 'dx', d.dx);
end

function tr = placeWavelet(tr, t, dt, w, amp)
% add amp*w centred at time t with linear sub-sample interpolation
nt = numel(tr);
half = floor(numel(w) / 2);
pos = t / dt;
i0 = floor(pos);
frac = pos - i0;
shifts = [0 1];
weights = [1 - frac, frac];
for s = 1:2
    c = i0 + shifts(s);               % 0-based centre sample
    lo = c - half;                     % 0-based first sample
    a = max(lo, 0);
    b = min(lo + numel(w), nt);        % exclusive
    if a < b
        tr(a+1:b) = tr(a+1:b) + amp * weights(s) * w(a-lo+1:b-lo);
    end
end
end
