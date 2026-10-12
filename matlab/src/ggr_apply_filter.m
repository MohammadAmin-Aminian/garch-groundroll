function filtered = ggr_apply_filter(data, dt, dx, fcfg, fkMask)
%GGR_APPLY_FILTER Conventional ground-roll filter described by a config struct.
%   fcfg.type: 'ormsby' (corners, mode), 'fk_fan' (v_reject, v_pass, f_max,
%   f_taper) or 'fk_mask' (needs FKMASK).
switch fcfg.type
    case 'ormsby'
        mode = 'bandstop';
        if isfield(fcfg, 'mode') && ~isempty(fcfg.mode), mode = fcfg.mode; end
        filtered = ggr_ormsby(data, dt, fcfg.corners, mode);
    case 'fk_fan'
        filtered = ggr_fk_fan(data, dt, dx, fcfg.v_reject, getf(fcfg, 'v_pass'), ...
            getf(fcfg, 'f_max'), getf(fcfg, 'f_taper'));
    case 'fk_mask'
        if nargin < 5 || isempty(fkMask)
            error('ggr:filter', 'filter type fk_mask needs the dataset''s f-k mask');
        end
        filtered = ggr_fk_mask(data, fkMask);
    otherwise
        error('ggr:filter', 'Unknown filter type %s', fcfg.type);
end
end

function v = getf(s, f)
if isfield(s, f), v = s.(f); else, v = []; end
end
