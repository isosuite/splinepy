"""
iges io.
"""

from datetime import datetime as _datetime

import numpy as _np

import splinepy as _splinepy


def _format_hollerith(s):
    return f"{len(s)}H{s}"


def _make_block(s, width=72, align_right=False):
    if len(s) > width:
        s = s[:width]
    return s.rjust(width) if align_right else s.ljust(width)


def _write_section(lines, content_str, section_id):
    chunk_size = 72
    chunks = [
        content_str[i : i + chunk_size] for i in range(0, len(content_str), chunk_size)
    ]
    if not chunks:
        chunks = [""]
    for i, chunk in enumerate(chunks):
        seq = i + 1
        line = _make_block(chunk, 72, False) + f"{section_id}{seq:7d}"
        lines.append(line)
    return len(chunks)


def _export_internal(fname, splines):
    if not isinstance(splines, (list, tuple)):
        splines = [splines]

    d_lines = []
    p_lines = []

    p_line_num = 1
    d_line_num = 1

    for s in splines:
        # Bump to 3D if needed
        cps = _np.array(s.control_points, dtype=_np.float64)
        if cps.shape[1] < 3:
            cps = _np.hstack([cps, _np.zeros((len(cps), 3 - cps.shape[1]))])

        is_rat = s.is_rational
        weights = (
            _np.array(s.weights, dtype=_np.float64).flatten()
            if is_rat
            else _np.ones(len(cps))
        )

        kvs = s.knot_vectors
        if kvs is None or len(kvs) == 0:
            kvs = []
            for d in s.degrees:
                kvs.append(_np.concatenate([_np.zeros(d + 1), _np.ones(d + 1)]))

        degs = list(s.degrees)

        tokens = []
        if s.para_dim == 1:
            entity_type = 126
            k = len(cps) - 1
            m = degs[0]
            tokens.extend([126, k, m, 0, 0, 0 if is_rat else 1, 0])
            # Knot vector
            tokens.extend(list(kvs[0]))
            # Weights
            tokens.extend(list(weights))
            # Control points
            for cp in cps:
                tokens.extend([cp[0], cp[1], cp[2]])
            # Parameter range
            tokens.extend([kvs[0][m], kvs[0][k + 1], 0.0, 0.0, 0.0])

        elif s.para_dim == 2:
            entity_type = 128
            k1 = s.control_mesh_resolutions[0] - 1
            k2 = s.control_mesh_resolutions[1] - 1
            m1 = degs[0]
            m2 = degs[1]
            tokens.extend([128, k1, k2, m1, m2, 0, 0, 0 if is_rat else 1, 0, 0])
            # Knot vectors
            tokens.extend(list(kvs[0]))
            tokens.extend(list(kvs[1]))
            # Weights
            tokens.extend(list(weights))
            # Control points
            for cp in cps:
                tokens.extend([cp[0], cp[1], cp[2]])
            # Parameter range
            tokens.extend([kvs[0][m1], kvs[0][k1 + 1], kvs[1][m2], kvs[1][k2 + 1]])
        else:
            raise ValueError(
                f"IGES only supports 1D or 2D splines, given {s.para_dim}D."
            )

        p_str = (
            ",".join(
                str(int(t)) if isinstance(t, (int, _np.integer)) else f"{t:.16g}"
                for t in tokens
            )
            + ";"
        )

        # P section uses columns 1-64 for data, 65-72 for the directory
        # pointer, and 73-80 for the sequence number.
        chunk_size = 64
        p_chunks = [p_str[i : i + chunk_size] for i in range(0, len(p_str), chunk_size)]
        p_start = p_line_num
        p_count = len(p_chunks)

        for chunk in p_chunks:
            p_line = _make_block(chunk, 64, False) + f"{d_line_num:8d}P{p_line_num:7d}"
            p_lines.append(p_line)
            p_line_num += 1

        # Directory entry line 1 & 2
        d1 = (
            f"{entity_type:8d}{p_start:8d}{0:8d}{1:8d}{0:8d}{0:8d}{0:8d}"
            f"{0:8d}00010001D{d_line_num:7d}"
        )
        d2 = (
            f"{entity_type:8d}{0:8d}{0:8d}{p_count:8d}{0:8d}{0:8d}{0:8d}"
            f"{'':8s}{0:8d}D{d_line_num + 1:7d}"
        )
        d_lines.append(d1)
        d_lines.append(d2)
        d_line_num += 2

    # Global section
    now = _datetime.now().strftime("%Y%m%d.%H%M%S")
    g_tokens = [
        _format_hollerith(","),
        _format_hollerith(";"),
        _format_hollerith("splinepy"),
        _format_hollerith(fname),
        _format_hollerith("splinepy"),
        _format_hollerith("splinepy"),
        "32",
        "38",
        "6",
        "308",
        "15",
        _format_hollerith("receiver"),
        "1.0",
        "6",
        _format_hollerith("M"),
        "1",
        "0.0024384",
        _format_hollerith(now),
        "0.00000254",
        "1000.0",
        _format_hollerith("Author"),
        _format_hollerith("splinepy"),
        "11",
        "0",
        _format_hollerith(now),
        _format_hollerith("NULL"),
    ]
    g_str = ",".join(g_tokens) + ";"
    g_chunks = [g_str[i : i + 72] for i in range(0, len(g_str), 72)]
    g_lines = [
        _make_block(chunk, 72, False) + f"G{i + 1:7d}"
        for i, chunk in enumerate(g_chunks)
    ]

    s_lines = [_make_block("IGES file generated by splinepy", 72, False) + f"S{1:7d}"]

    s_count = len(s_lines)
    g_count = len(g_lines)
    d_count = len(d_lines)
    p_count = len(p_lines)

    t_line = (
        f"S{s_count:7d}G{g_count:7d}D{d_count:7d}P{p_count:7d}".ljust(72) + f"T{1:7d}"
    )

    all_lines = s_lines + g_lines + d_lines + p_lines + [t_line]
    with open(fname, "w") as f:
        f.writelines(line + "\n" for line in all_lines)


def _load_internal(fname):
    with open(fname) as f:
        lines = [line for line in f if len(line) > 73]
    # column 73 names the section
    d_lines = [line for line in lines if line[72] == "D"]
    p_lines = [line for line in lines if line[72] == "P"]

    # parameter text by sequence number; an entity's lines end with ;
    p_data_by_line = {int(line[73:80]): line[:64] for line in p_lines}

    splines = []
    # Entities in D section come in pairs
    for d1, d2 in zip(d_lines[::2], d_lines[1::2], strict=True):
        entity_type = int(d1[:8])
        if entity_type not in (126, 128):
            continue
        p_start = int(d1[8:16])
        # 126 (curve): type, K, M, 4 flags; 128 (surface): type, K1, K2, M1,
        # M2, 5 flags. Then knots per axis, weights and control points.
        n_axes = 1 if entity_type == 126 else 2
        *head, numbers = "".join(
            p_data_by_line[p_start + j] for j in range(int(d2[24:32]))
        ).split(",", 4 + 3 * n_axes)
        upper = [int(t) for t in head[1 : 1 + n_axes]]
        degrees = [int(t) for t in head[1 + n_axes : 1 + 2 * n_axes]]
        is_rational = int(head[3 + 2 * n_axes]) == 0
        n_knots = [k + m + 2 for k, m in zip(upper, degrees, strict=True)]
        n_cps = int(_np.prod([k + 1 for k in upper]))
        numbers = numbers.partition(";")[0].split(",")
        values = _np.array(
            [float(t) for t in numbers[: sum(n_knots) + 4 * n_cps]]
        )
        properties = {
            "degrees": degrees,
            "knot_vectors": _np.split(values[: sum(n_knots)], n_knots[:-1]),
            "control_points": values[sum(n_knots) + n_cps :].reshape(-1, 3),
        }
        if is_rational:
            weights = values[sum(n_knots) : sum(n_knots) + n_cps]
            splines.append(_splinepy.NURBS(weights=weights, **properties))
        else:
            splines.append(_splinepy.BSpline(**properties))

    return splines


def load(fname):
    """
    Read spline in `.iges` form.

    Parameters
    -----------
    fname: str

    Returns
    --------
    splines: list
      Spline Type defined in NAME_TO_TYPE
    """
    return _load_internal(fname)


def export(fname, splines):
    """
    Save splines as `.iges`.

    Parameters
    -----------
    fname: str
    spline: list
      list of Splines

    Returns
    --------
    None
    """
    return _export_internal(fname, splines)
