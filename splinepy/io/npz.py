"""
npz io.

keys in raw files are:
  - control_points : (ncpts, dim) np.ndarray, float64
  - degrees : (para_dim,) np.ndarray, int32
  - knot_vectors : (1,) np.ndarray, str
  - whatami : (1,) np.ndarray, str
"""

import numpy as _np

from splinepy import io as _io


def load(
    fname,
):
    """
    Read a list of splines in `.npz` form.

    Parameters
    -----------
    fname: str

    Returns
    --------
    list_spline: list of NURBS / BSpline / Bezier / Rational Bezier
    """
    list_of_spline_dicts = []
    with _np.load(fname) as loaded:
        # Loop through keys in loaded dictionary and find 'spline_'
        for i in range(loaded["number_of_splines"]):
            dict_spline = {
                "control_points": loaded[f"spline_{i}_control_points"],
                "degrees": loaded[f"spline_{i}_degrees"],
            }
            if f"spline_{i}_weights" in loaded:
                dict_spline["weights"] = loaded[f"spline_{i}_weights"]
            if f"spline_{i}_knot_vectors_0" in loaded:
                dict_spline["knot_vectors"] = [
                    loaded[f"spline_{i}_knot_vectors_{j}"]
                    for j in range(dict_spline["degrees"].size)
                ]
            list_of_spline_dicts.append(dict_spline)

    return _io.ioutils.dict_to_spline(list_of_spline_dicts)


def export(fname, list_of_splines):
    """
    Save a list of splines as `.npz`.

    Parameters
    -----------
    fname: str
    splines: list of NURBS / BSpline / Bezier / Rational Bezier

    Returns
    --------
    None
    """

    # Checking for proper type of input
    if not isinstance(list_of_splines, list):
        from splinepy.spline import Spline as _Spline

        if isinstance(list_of_splines, _Spline):
            list_of_splines = [list_of_splines]
        else:
            raise TypeError("Only a list of splines or a spline can be saved.")

    # Initialize an empty dictionary to store the properties of each spline
    property_dicts = {}

    # Add number of splines in list to dict
    property_dicts["number_of_splines"] = len(list_of_splines)

    # Loop through the list of splines and add their properties to the dictionary
    for i, spline in enumerate(list_of_splines):
        # Use a prefix to distinguish different splines
        prefix = f"spline_{i}_"

        # Add the common properties of all splines
        property_dicts[f"{prefix}degrees"] = spline.degrees
        property_dicts[f"{prefix}control_points"] = spline.control_points

        # Add the weights if the spline is rational
        if spline.is_rational:
            property_dicts[f"{prefix}weights"] = spline.weights

        # Add the knot vectors if the spline has them
        if spline.has_knot_vectors:
            for j in range(spline.para_dim):
                property_dicts[f"{prefix}knot_vectors_{j}"] = (
                    spline.knot_vectors[j]
                )

    # Save the dictionary as `.npz`
    _np.savez(
        fname,
        **property_dicts,
    )
