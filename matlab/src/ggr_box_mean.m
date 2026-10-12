function m = ggr_box_mean(A, sz)
%GGR_BOX_MEAN Moving average with edge replication ('nearest').
%   m = GGR_BOX_MEAN(A, [h w]) averages A over an h x w box. For even sizes
%   the box covers [i-h/2, i+h/2-1], matching scipy.ndimage.uniform_filter.
if isscalar(sz), sz = [sz sz]; end
[nr, nc] = size(A);
ri = clampIdx((1:nr)', sz(1), nr);
ci = clampIdx((1:nc)', sz(2), nc);
B = A(ri, ci);
m = conv2(ones(sz(1), 1) / sz(1), ones(1, sz(2)) / sz(2), B, 'valid');
end

function idx = clampIdx(i, s, n)
lo = floor(s / 2);
hi = s - lo - 1;
idx = min(max((i(1) - lo):(i(end) + hi), 1), n);
end
