function out = ggr_fk_mask(d, noiseMask)
%GGR_FK_MASK f-k filter with an explicit reject mask on the centred spectrum.
%
%   out = GGR_FK_MASK(d, mask) returns the filtered data, where MASK (same
%   size as D) is 1 on the region of fftshift(fft2(d)) to remove. This is
%   the original workflow
%       noise = real(ifft2(ifftshift(fftshift(fft2(d)) .* mask)));
%       out   = d - noise;
%   without the interactive ROIPOLY step.

d = double(d);
noiseMask = double(noiseMask);
if ~isequal(size(noiseMask), size(d))
    error('ggr:fkmask', 'mask must have the same size as d');
end
S = fftshift(fft2(d));
noise = real(ifft2(ifftshift(S .* noiseMask)));
out = d - noise;
end
