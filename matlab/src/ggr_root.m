function r = ggr_root()
%GGR_ROOT Repository root folder (contains data/ and configs/).
r = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
