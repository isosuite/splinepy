"""Translates gustaf / vedo style show options to viewer arguments."""

import numpy as _np

from splinepy.utils import log as _log

_INDEXED_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)  # fmt: skip

ALIASES = {
    "c": "color",
    "alpha": "opacity",
    "cmap": "colormap",
    "data": "scalars",
    "arrow_data": "arrows",
    "arrow_data_on": "arrow_on",
    "arrow_data_scale": "arrow_scale",
    "arrow_data_color": "arrow_color",
    "arrow_data_to_origin": "arrow_to_origin",
    "knot_c": "knot_color",
    "knot_lw": None,
    "knot_alpha": None,
    "lw": "line_width",
    "control_point_c": "control_point_color",
    "control_point_r": "control_point_size",
    "control_mesh_c": "control_mesh_color",
    "control_mesh_lw": None,
    "control_point_alpha": None,
    "resolution": "resolutions",
    "res": "resolutions",
    "r": "control_point_size",
    "cam": "camera",
}
# None marks an accepted vedo option with no viewer equivalent.

IGNORED = {
    "vertex_ids", "element_ids", "scalarbar", "scalarbar3d", "cmap_alpha",
    "cmap_n_colors", "labels", "label_options", "parametric_space",
    "return_gustaf", "return_showable", "backend", "open", "block", "new_window",
    "interactive", "title",
    "close", "N", "sz", "size", "camera_options",
}  # fmt: skip
# These Gustaf and vedo options do not affect the viewer.


def to_css(c):
    """Convert a color name, RGB(A) value, or integer index to CSS.

    Parameters
    ----------
    c : int, str or array-like
        Palette index, CSS color, or RGB components.

    Returns
    -------
    color : str or None
        CSS color, or None for an unsupported value.
    """
    if isinstance(c, (int, _np.integer)) and not isinstance(c, bool):
        return _INDEXED_COLORS[abs(int(c)) % len(_INDEXED_COLORS)]
    if isinstance(c, str):
        return c
    if isinstance(c, (tuple, list, _np.ndarray)) and len(c) in (3, 4):
        rgb = _np.asarray(c[:3], dtype=float)
        if rgb.max() <= 1:
            rgb = rgb * 255
        return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)
    return None


def translate(kwargs):
    """Returns viewer keyword arguments for vedo style ``kwargs``.

    Old names are renamed (see :data:`ALIASES`), colors are converted to css,
    options without equivalent are dropped with a debug message. Everything
    else is passed through unchanged, so typos still raise in ``show``.

    Parameters
    ----------
    kwargs : dict
        Visualization options, including legacy aliases.

    Returns
    -------
    options : dict
        Options using viewer keyword names.
    """
    from splinepy.viewer.scene import COLORMAPS, LIGHTING

    out = {}
    for original_key, original_value in kwargs.items():
        key, value = original_key, original_value
        if key in ALIASES:
            new = ALIASES[key]
            if new is None:
                _log.debug(f"Viewer: option '{key}' has no equivalent.")
                continue
            if new in kwargs:
                continue  # explicit new name wins
            key = new
        elif key in IGNORED:
            _log.debug(f"Viewer: ignoring option '{key}'.")
            continue

        if key.endswith("color"):
            css = to_css(value)
            if css is None:
                _log.debug(f"Viewer: can't use '{key}'={value!r} as color.")
                continue
            value = css
        elif key == "colormap" and value not in COLORMAPS:
            _log.warning(
                f"Viewer: colormap '{value}' unknown, using viridis. "
                f"Available: {COLORMAPS}"
            )
            value = "viridis"
        elif key == "axes" and not isinstance(value, bool):
            value = True  # vedo axes dict
        elif key == "lighting" and value not in (None, *LIGHTING):
            _log.warning(f"Viewer: lighting '{value}' unknown, using default.")
            value = "default"
        out[key] = value
    return out
