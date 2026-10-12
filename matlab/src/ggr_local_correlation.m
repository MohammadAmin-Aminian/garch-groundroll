function c = ggr_local_correlation(a, b, win)
%GGR_LOCAL_CORRELATION Local Pearson correlation in a win x win box (default 9).
if nargin < 3 || isempty(win), win = 9; end
a = double(a); b = double(b);
ma = ggr_box_mean(a, win);
mb = ggr_box_mean(b, win);
cv = ggr_box_mean(a .* b, win) - ma .* mb;
va = max(ggr_box_mean(a .^ 2, win) - ma .^ 2, 0);
vb = max(ggr_box_mean(b .^ 2, win) - mb .^ 2, 0);
den = sqrt(va .* vb);
c = zeros(size(cv));
ok = den > 1e-12 * (max(den(:)) + 1e-30);
c(ok) = cv(ok) ./ den(ok);
c = min(max(c, -1), 1);
end
