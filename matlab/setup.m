%SETUP Add the toolbox to the MATLAB path (run once per session).
%   >> run('matlab/setup.m')
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'src'));
fprintf('garch-groundroll MATLAB toolbox added to the path (%s)\n', fullfile(here, 'src'));
if verLessThan('matlab', '9.1')
    warning('MATLAB R2016b or newer is required (implicit expansion, jsondecode).');
end
