function g = ggr_gabor_kernel(wavelength, orientationDeg, bandwidth, aspectRatio)
%GGR_GABOR_KERNEL Complex Gabor kernel (definition of MATLAB's GABOR object).
%
%   g = GGR_GABOR_KERNEL(lambda, theta) with wavelength LAMBDA (pixels per
%   cycle) and orientation THETA (degrees, counter-clockwise from the
%   column axis, rows pointing down). Spatial-frequency bandwidth 1 octave
%   and spatial aspect ratio 0.5 by default. Kernel half-width ceil(3*sigma).

if nargin < 3 || isempty(bandwidth), bandwidth = 1; end
if nargin < 4 || isempty(aspectRatio), aspectRatio = 0.5; end
sigma = wavelength / pi * sqrt(log(2) / 2) * (2 ^ bandwidth + 1) / (2 ^ bandwidth - 1);
sx = sigma;
sy = sigma / aspectRatio;
r = ceil(3 * max(sx, sy));
[x, y] = meshgrid(-r:r, -r:r);
th = orientationDeg * pi / 180;
xr = x * cos(th) - y * sin(th);
yr = x * sin(th) + y * cos(th);
env = exp(-0.5 * (xr .^ 2 / sx ^ 2 + yr .^ 2 / sy ^ 2));
g = env .* exp(1i * 2 * pi * xr / wavelength);
end
