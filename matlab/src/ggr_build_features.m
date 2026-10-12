function X = ggr_build_features(csd, wavelengths, orientations, smoothing, includeCoords, scaling)
%GGR_BUILD_FEATURES Standardised texture features of the CSD image.
%
%   X = GGR_BUILD_FEATURES(csd, wavelengths, orientations, smoothing,
%   includeCoords, scaling) returns an (nt*nx) x nFeatures matrix (pixels
%   in column-major order, i.e. csd(:)). Features: the scaled CSD, the
%   Gabor magnitude for every wavelength/orientation pair (each smoothed by
%   a Gaussian with sigma = smoothing*wavelength), and optionally the column
%   and row index. Each feature is standardised to zero mean, unit variance.
%
%   SCALING: 'clip' (min(max(csd,0),1), as im2uint16 in the original study),
%   'minmax' or 'none'. No toolbox is required.

if nargin < 2 || isempty(wavelengths),  wavelengths = [3 6 12 24 48 96]; end
if nargin < 3 || isempty(orientations), orientations = [0 45 90 135]; end
if nargin < 4 || isempty(smoothing),    smoothing = 1.5; end
if nargin < 5 || isempty(includeCoords), includeCoords = true; end
if nargin < 6 || isempty(scaling),      scaling = 'clip'; end

img = double(csd);
switch lower(scaling)
    case 'clip',   img = min(max(img, 0), 1);
    case 'minmax'
        lo = min(img(:)); hi = max(img(:));
        if hi > lo, img = (img - lo) / (hi - lo); else, img = zeros(size(img)); end
    case 'none'
    otherwise, error('ggr:scaling', 'scaling must be clip, minmax or none');
end
[nt, nx] = size(img);
nF = 1 + numel(wavelengths) * numel(orientations) + 2 * logical(includeCoords);
X = zeros(nt * nx, nF);
X(:, 1) = img(:);
c = 1;
for lam = wavelengths(:)'
    for th = orientations(:)'
        g = ggr_gabor_kernel(lam, th);
        r = (size(g, 1) - 1) / 2;
        P = img(symIdx(nt, r), symIdx(nx, r));
        mag = abs(fftconvSame(P, g));
        mag = mag(r+1:r+nt, r+1:r+nx);
        if smoothing > 0
            mag = gaussSmooth(mag, smoothing * lam);
        end
        c = c + 1;
        X(:, c) = mag(:);
    end
end
if includeCoords
    [C, R] = meshgrid(0:nx-1, 0:nt-1);
    X(:, c + 1) = C(:);
    X(:, c + 2) = R(:);
end
mu = mean(X, 1);
sd = std(X, 1, 1);
sd(sd == 0) = 1;
X = (X - mu) ./ sd;
end

% -------------------------------------------------------------------------
function idx = symIdx(n, r)
% indices of a symmetric (half-sample) padding by r on both sides, any r
i = (-r:n-1+r)';
i = mod(i, 2 * n);
i(i >= n) = 2 * n - 1 - i(i >= n);
idx = i + 1;
end

function out = fftconvSame(A, K)
% 2-D convolution, output the same size as A (centre of the full result)
[ma, na] = size(A);
[mk, nk] = size(K);
m = ma + mk - 1;
n = na + nk - 1;
F = ifft2(fft2(A, m, n) .* fft2(K, m, n));
r0 = floor((mk - 1) / 2);
c0 = floor((nk - 1) / 2);
out = F(r0+1:r0+ma, c0+1:c0+na);
end

function B = gaussSmooth(A, sigma)
% Gaussian filter, truncate at 2 sigma, edge replication ('nearest')
r = floor(2 * sigma + 0.5);
x = (-r:r)';
k = exp(-0.5 * (x / sigma) .^ 2);
k = k / sum(k);
[nr, nc] = size(A);
ri = min(max((1-r):(nr+r), 1), nr);
ci = min(max((1-r):(nc+r), 1), nc);
B = conv2(k, k', A(ri, ci), 'valid');
end
