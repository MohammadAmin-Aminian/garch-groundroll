function cfg = ggr_load_config(nameOrPath)
%GGR_LOAD_CONFIG Read configs/<name>.json (shared with the Python package)
%   and merge it into the defaults. Requires JSONDECODE (MATLAB R2016b+).
if endsWith(nameOrPath, '.json')
    p = nameOrPath;
else
    p = fullfile(ggr_root(), 'configs', [nameOrPath '.json']);
end
user = jsondecode(fileread(p));
cfg = ggr_merge_config(ggr_default_config(), user);
end
