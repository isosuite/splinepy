"""Camera helpers for scripted views and animations."""

import numpy as _np


def _bounds(scene):
    if "panels" in scene:
        scene = scene["panels"][0]
    return _np.asarray(scene["options"]["bounds"], dtype=float)


def turntable(
    scene, n=60, elevation=25.0, distance=None, up_axis="y", turns=1.0, start=45.0
):
    """Cameras orbiting the scene center, e.g. for a rotating video.

    Parameters
    ----------
    n: int
        Number of frames.
    elevation: float
        Degrees above the horizontal plane.
    distance: float
        Camera distance from center. Default: 1.6 x bounding box diagonal.
    up_axis: str
        "y" or "z".
    turns: float
        Number of full revolutions over the n frames (endpoint excluded, so
        a full turn loops seamlessly).
    start: float
        Starting azimuth in degrees.
    """
    lo, hi = _bounds(scene)
    center = (lo + hi) / 2
    distance = distance or 1.6 * float(_np.linalg.norm(hi - lo))
    el = _np.radians(elevation)
    az = _np.radians(start) + 2 * _np.pi * turns * _np.arange(n) / n
    h = _np.cos(el) * distance
    cams = []
    for a in az:
        if up_axis == "y":
            off = [h * _np.cos(a), _np.sin(el) * distance, h * _np.sin(a)]
            up = [0, 1, 0]
        elif up_axis == "z":
            off = [h * _np.cos(a), h * _np.sin(a), _np.sin(el) * distance]
            up = [0, 0, 1]
        else:
            raise ValueError("up_axis must be 'y' or 'z'")
        cams.append(
            {
                "position": (center + off).tolist(),
                "target": center.tolist(),
                "up": up,
            }
        )
    return cams
