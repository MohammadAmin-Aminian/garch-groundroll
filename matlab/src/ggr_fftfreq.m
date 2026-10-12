function f = ggr_fftfreq(n, d)
%GGR_FFTFREQ Sample frequencies of an n-point FFT (same order as fft), column.
f = [0:ceil(n/2)-1, -floor(n/2):-1]' / (d * n);
end
