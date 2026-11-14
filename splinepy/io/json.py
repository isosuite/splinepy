"""
Export splines in custom json format
"""

import base64 as _base64
import json as _json

import numpy as _np

from splinepy import settings as _settings
from splinepy.utils.log import debug as _debug


def _hierarchy_types():
    from splinepy.hierarchy import (
        HBSpline,
        RationalHBSpline,
        RationalTHBSpline,
        THBSpline,
    )

    return {
        cls.__name__: cls
        for cls in (HBSpline, RationalHBSpline, THBSpline, RationalTHBSpline)
    }


def _hierarchy_record(spline):
    data, cache = spline._state
    degrees, levels, _, _, points, weights, _ = data
    record = {
        "degrees": degrees.tolist(),
        "levels": [[axis.tolist() for axis in level] for level in levels],
        "leaf_cells": spline._cells(degrees, levels, cache.extraction()).tolist(),
        "control_points": points.tolist(),
    }
    if weights is not None:
        record["weights"] = weights.tolist()
        record["weights_ndim"] = weights.ndim
    return record


def load(fname):
    """
    Loads splines from json file. Returns a list of splines

    Parameters
    -----------
    fname: str

    Returns
    --------
    spline_list: list
    """
    # Make Required Properties locally accessible
    from splinepy.spline import RequiredProperties as _RequiredProperties

    # Import data from file into dict format
    with open(fname) as f:
        jsonbz = _json.load(f)

    spline_list = []
    base64encoding = jsonbz["Base64Encoding"]
    hierarchy_types = _hierarchy_types()

    for jbz in jsonbz["SplineList"]:
        hierarchy_type = hierarchy_types.get(jbz["SplineType"])
        # Convert Base64 str into np array
        if base64encoding:
            jbz["control_points"] = _np.frombuffer(
                _base64.b64decode(jbz["control_points"].encode("ascii")),
                dtype=_np.float64,
            ).reshape((-1, jbz["dim"]))
            if jbz.get("weights") is not None:
                jbz["weights"] = _np.frombuffer(
                    _base64.b64decode(jbz["weights"].encode("ascii")),
                    dtype=_np.float64,
                ).reshape((-1, 1))
                if hierarchy_type is not None and jbz.get("weights_ndim") == 1:
                    jbz["weights"] = jbz["weights"].ravel()

        if hierarchy_type is not None:
            spline_list.append(
                hierarchy_type.from_hierarchy(
                    jbz["degrees"],
                    jbz["levels"],
                    jbz["leaf_cells"],
                    jbz["control_points"],
                    weights=jbz.get("weights"),
                )
            )
            continue

        # Declare Type and get required properties
        req_props = _RequiredProperties.of(jbz["SplineType"])
        data_dict = {}
        for prop in req_props:
            data_dict[prop] = jbz[prop]
        spline_list.append(_settings.NAME_TO_TYPE[jbz["SplineType"]](**data_dict))

        # Add show options if any
        show_options = jbz.get("show_options", None)
        if show_options is not None:
            for key, value in show_options.items():
                spline_list[-1].show_options[key] = value

    _debug("Imported " + str(len(spline_list)) + " splines from file.")

    return spline_list


def export(
    fname, spline_list, list_name=None, base64encoding=False, *, extraction_basis=None
):
    """
    Exports a list of arbitrary splines in json-format

    Parameters
    ----------
    fname : str
      Export Filename
    spline_list: list
      List of arbitrary Spline-Types
    list_name: str
      Default is None and "SplineGroup" will be assigned. Used to define name.
    base64encoding: bool
      Default is False. If True, encodes float type spline properties before
      saving.
    extraction_basis : {"bspline", "bernstein"} or None
      Include optional leaf extraction blocks in each hierarchical spline
      record. Loading reconstructs the hierarchy from its geometry and topology.

    Returns
    -------
    output_dict : dict
      Dictornay data written into file

    """
    # Determine Spline Group
    if list_name is None:
        list_name = "SplineGroup"

    # Ensure spline_list is actually a list
    if not isinstance(spline_list, list):
        spline_list = [spline_list]

    # Number of splines in group
    n_splines = len(spline_list)

    output_dict = {
        "Name": list_name,
        "NumberOfSplines": n_splines,
        "Base64Encoding": base64encoding,
    }

    # Append all splines to a dictionary
    from splinepy.spline import Spline as _Spline

    hierarchy_types = tuple(_hierarchy_types().values())
    spline_dictonary_list = []
    for i_spline in spline_list:
        is_hierarchy = type(i_spline) in hierarchy_types
        if is_hierarchy:
            i_spline_dict = _hierarchy_record(i_spline)
            if extraction_basis is not None:
                extraction = i_spline.extract.basis_data(extraction_basis)
                i_spline_dict["extraction"] = {
                    key: value.tolist() if isinstance(value, _np.ndarray) else value
                    for key, value in extraction.items()
                }
        elif isinstance(i_spline, _Spline):
            i_spline_dict = i_spline.todict(tolist=True)
        else:
            raise ValueError("Unsupported type in list")
        i_spline_dict["SplineType"] = type(i_spline).__qualname__
        # Append para_dim and dim
        i_spline_dict["dim"] = i_spline.dim
        i_spline_dict["para_dim"] = i_spline.para_dim

        if base64encoding:
            i_spline_dict["control_points"] = _base64.b64encode(
                _np.array(i_spline_dict["control_points"], dtype=_np.float64)
            ).decode("utf-8")
            if "weights" in i_spline_dict:
                i_spline_dict["weights"] = _base64.b64encode(
                    _np.array(i_spline_dict["weights"], dtype=_np.float64)
                ).decode("utf-8")

        if not is_hierarchy:
            i_spline_dict["show_options"] = dict(i_spline.show_options)

        spline_dictonary_list.append(i_spline_dict)

    output_dict["SplineList"] = spline_dictonary_list

    with open(fname, "w") as file_pointer:
        file_pointer.write(_json.dumps(output_dict, indent=4))

    return output_dict
