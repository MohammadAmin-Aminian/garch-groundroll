function ds = ggr_load_dataset(nameOrPath)
%GGR_LOAD_DATASET Load data/<name>.mat (or a .mat path).
%   Fields: name, data, dt, dx, description, fkMask ([] if absent).
if endsWith(nameOrPath, '.mat')
    p = nameOrPath;
else
    p = fullfile(ggr_root(), 'data', [nameOrPath '.mat']);
end
if ~exist(p, 'file'), error('ggr:data', 'Dataset not found: %s', p); end
s = load(p);
[~, name] = fileparts(p);
ds.name = name;
ds.data = double(s.data);
ds.dt = double(s.dt);
ds.dx = double(s.dx);
ds.description = '';
if isfield(s, 'description'), ds.description = strtrim(char(s.description)); end
ds.fkMask = [];
if isfield(s, 'fk_mask'), ds.fkMask = double(s.fk_mask); end
end
