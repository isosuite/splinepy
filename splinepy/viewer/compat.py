"""Translates the short show options to viewer arguments."""

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
    "knot_lw": "knot_line_width",
    "knot_alpha": "knot_opacity",
    "lw": "line_width",
    "control_point_c": "control_point_color",
    "control_point_r": "control_point_size",
    "control_mesh_c": "control_mesh_color",
    "control_mesh_lw": "control_mesh_line_width",
    "control_mesh_alpha": "control_mesh_opacity",
    "control_point_alpha": "control_point_opacity",
    "resolution": "resolutions",
    "res": "resolutions",
    "r": "control_point_size",
    "cam": "camera",
}

# Options without a viewer equivalent.
IGNORED = {
    "vertex_ids", "element_ids", "scalarbar", "scalarbar3d", "cmap_alpha",
    "cmap_n_colors", "labels", "label_options", "parametric_space",
    "open", "block", "new_window",
    "interactive", "title",
    "close", "N", "sz", "size", "camera_options",
}  # fmt: skip


# matplotlib one-letter color names
_LETTERS = {
    "k": "black", "w": "white", "r": "red", "g": "green", "b": "blue",
    "c": "cyan", "m": "magenta", "y": "yellow",
}  # fmt: skip

# Color stops of the colormaps, evenly spaced. viewer.js has the same table.
COLORMAP_STOPS = {
    "viridis": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
    "plasma": ("#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"),
    "coolwarm": ("#3b4cc0", "#8db0fe", "#dddddd", "#f4987a", "#b40426"),
    "jet": ("#000080", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#800000"),
    "gray": ("#000000", "#ffffff"),
}


def colormap(values, name="plasma", vmin=None, vmax=None, n_colors=None):
    """Maps values to RGB colors, like the viewer does.

    Parameters
    ----------
    values : array-like
        Scalar values.
    name : str, optional
        One of :data:`COLORMAP_STOPS`. Unknown names use plasma.
    vmin, vmax : float, optional
        Value range mapped to the ends of the colormap. Default: data range.
    n_colors : int, optional
        Number of discrete colors. Default: continuous.

    Returns
    -------
    rgb : (n, 3) numpy.ndarray
        Colors with components in [0, 1].
    """
    if name not in COLORMAP_STOPS:
        _log.warning(f"colormap '{name}' unknown, using plasma.")
        name = "plasma"
    values = _np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return _np.empty((0, 3), dtype=float)
    lo = values.min() if vmin is None else vmin
    hi = values.max() if vmax is None else vmax
    t = _np.clip((values - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    if n_colors:
        t = _np.minimum(_np.floor(t * n_colors), n_colors - 1) / max(n_colors - 1, 1)
    stops = _np.array(
        [[int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)] for h in COLORMAP_STOPS[name]]
    )
    x = t * (len(stops) - 1)
    i = _np.minimum(x.astype(int), len(stops) - 2)
    w = (x - i)[:, None]
    return stops[i] * (1 - w) + stops[i + 1] * w


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
        return _LETTERS.get(c, c)
    if isinstance(c, (tuple, list, _np.ndarray)) and len(c) in (3, 4):
        rgb = _np.asarray(c[:3], dtype=float)
        if rgb.max() <= 1:
            rgb = rgb * 255
        rgb = _np.clip(rgb, 0, 255)
        return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)
    return None


def follow_control_points(options):
    """Hides control mesh and ids with the control points, unless given.

    Changes and returns ``options``.

    Parameters
    ----------
    options : dict
        Show options, after all of them are combined.

    Returns
    -------
    options : dict
        The same dict.
    """
    if not options.get("control_points", True):
        options.setdefault("control_mesh", False)
        options.setdefault("control_point_ids", False)
    return options


def translate(kwargs):
    """Returns viewer keyword arguments for short style ``kwargs``.

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
            if new in kwargs:
                continue  # explicit new name wins
            key = new
        elif key in IGNORED:
            _log.debug(f"Viewer: ignoring option '{key}'.")
            continue

        if key.endswith("color") or key == "background":
            css = to_css(value)
            if css is None:
                _log.debug(f"Viewer: can't use '{key}'={value!r} as color.")
                continue
            value = css
        elif key == "colormap" and value not in COLORMAPS:
            _log.warning(
                f"Viewer: colormap '{value}' unknown, using plasma. "
                f"Available: {COLORMAPS}"
            )
            value = "plasma"
        elif key == "axes" and not isinstance(value, bool):
            value = True  # a dict of axes options
        elif key == "lighting" and value not in (None, *LIGHTING):
            _log.warning(f"Viewer: lighting '{value}' unknown, using default.")
            value = "default"
        out[key] = value
    return out
