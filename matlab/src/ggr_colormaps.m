function [blues, series] = ggr_colormaps()
%GGR_COLORMAPS Single-hue blue ramp (magnitudes) and categorical colours.
anchors = [244 248 253; 205 226 251; 134 182 239; 57 135 229; 37 106 191; ...
    24 79 149; 13 54 107] / 255;
x = linspace(0, 1, size(anchors, 1));
blues = interp1(x, anchors, linspace(0, 1, 256));
series = [42 120 214; 235 104 52; 27 175 122; 237 161 0; 232 123 164; ...
    0 131 0; 98 80 214; 227 73 72] / 255;
end
