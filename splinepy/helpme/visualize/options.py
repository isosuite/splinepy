"""Show options shared by the viewer and the legacy Vedo backend."""

from copy import deepcopy as _deepcopy

import numpy as _np

_color = (str, tuple, list, int)
_number = (float, int)
_common_options = {
    "c": _color,
    "alpha": _number,
    "data": (str,),
    "vertex_ids": (bool,),
    "element_ids": (bool,),
    "lighting": (str,),
    "cmap": (str,),
    "vmin": _number,
    "vmax": _number,
    "cmap_alpha": _number,
    "cmap_n_colors": (int,),
    "scalarbar": (bool, dict),
    "scalarbar3d": (bool, dict),
    "arrow_data": (str,),
    "arrow_data_scale": _number,
    "arrow_data_color": _color,
    "arrow_data_to_origin": (bool,),
    "axes": (bool, dict),
    "knots": (bool,),
    "knot_c": _color,
    "knot_lw": _number,
    "knot_alpha": _number,
    "control_points": (bool,),
    "control_point_r": _number,
    "control_point_c": _color,
    "control_point_alpha": _number,
    "control_point_ids": (bool,),
    "control_mesh": (bool,),
    "control_mesh_c": _color,
    "control_mesh_lw": _number,
    "control_mesh_alpha": _number,
    "resolutions": (int, list, tuple, _np.ndarray),
    "lw": _number,
}


class _ShowOption:
    __slots__ = ("_helpee", "_options")

    _valid_options = _common_options

    def __init__(self, helpee):
        if self._helps not in str(type(helpee).__mro__):
            raise TypeError(f"These show options are for {self._helps}.")
        self._helpee = helpee
        self._options = {}

    def __setitem__(self, key, value):
        allowed = self._valid_options.get(key)
        if allowed is None:
            raise ValueError(f"{key} is an invalid option for {self._helps}.")
        if not isinstance(value, allowed):
            raise TypeError(f"{type(value)} is invalid for show option {key!r}.")
        self._options[key] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._options[key]
        return {name: self._options[name] for name in key if name in self._options}

    def __contains__(self, key):
        return key in self._options

    def __call__(self, **kwargs):
        self.update(**kwargs)
        return self._helpee

    def get(self, key, default=None):
        return self._options.get(key, default)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def valid_keys(self):
        return self._valid_options.keys()

    def keys(self):
        return self._options.keys()

    def values(self):
        return self._options.values()

    def items(self):
        return self._options.items()

    def clear(self):
        self._options.clear()

    def pop(self, *args):
        return self._options.pop(*args)

    def copy_valid_options(self, copy_to, keys=None):
        valid_keys = copy_to.valid_keys()
        options = self.items() if keys is None else self[keys].items()
        for key, value in options:
            if key in valid_keys:
                copy_to[key] = _deepcopy(value)


class SplineShowOption(_ShowOption):
    """Show options for splines."""

    __slots__ = ()
    _helps = "Spline"
    _valid_options = {
        **_common_options,
        "fitting_queries": (bool,),
        "arrow_data_on": (list, tuple, _np.ndarray),
    }
