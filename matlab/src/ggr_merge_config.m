function out = ggr_merge_config(base, override)
%GGR_MERGE_CONFIG Recursively overwrite fields of BASE with those of OVERRIDE.
out = base;
if isempty(override), return; end
fn = fieldnames(override);
for i = 1:numel(fn)
    f = fn{i};
    v = override.(f);
    if isstruct(v) && isfield(out, f) && isstruct(out.(f))
        out.(f) = ggr_merge_config(out.(f), v);
    else
        if isnumeric(v) && iscolumn(v), v = v'; end   % jsondecode returns columns
        out.(f) = v;
    end
end
end
