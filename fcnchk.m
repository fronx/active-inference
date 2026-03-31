function f = fcnchk(fun, varargin)
% Minimal Octave compatibility shim for MATLAB's fcnchk.
%
% SPM12 uses fcnchk during model setup. Octave may not provide it,
% which causes dynamic state functions to be discarded silently.

if isa(fun, 'function_handle') || isa(fun, 'inline')
    f = fun;
    return
end

if ischar(fun)
    if nargin > 1 && ~isempty(varargin)
        f = inline(fun, varargin{:});
    else
        f = str2func(fun);
    end
    return
end

error('Unsupported function specification for fcnchk');

end
