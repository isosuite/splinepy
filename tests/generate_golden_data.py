#!/usr/bin/env python3
"""
Generate comprehensive golden reference datasets from splinepy v0.2.1 oracle.
These datasets are used to verify the C++ core implementations in Phases 1-5
without requiring Python bindings in the C++ build.

Run from an isolated Python 3.12 environment with NumPy 2.2.6 and
splinepy 0.2.1 installed:

    python -I tests/generate_golden_data.py

The generated archives and raw binaries are local test data.
"""
from pathlib import Path
import numpy as np
import splinepy

OUT_DIR = Path(__file__).resolve().parent / "data" / "golden"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261027)

def make_queries(bounds, n=150):
    dim = bounds.shape[1]
    lows = bounds[0]
    highs = bounds[1]
    return RNG.uniform(lows, highs, size=(n, dim))

def generate_eval_bspline():
    data = {}
    
    # 1D Curve in 2D
    bs1 = splinepy.BSpline(
        degrees=[2],
        knot_vectors=[[0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0]],
        control_points=[[0.0, 0.0], [1.0, 1.5], [2.0, 0.5], [3.0, 2.0], [4.0, 0.0]],
    )
    q1 = make_queries(bs1.parametric_bounds)
    data["bs1_degrees"] = np.array(bs1.degrees, dtype=np.int32)
    data["bs1_kv_0"] = np.array(bs1.knot_vectors[0], dtype=np.float64)
    data["bs1_cps"] = np.array(bs1.control_points, dtype=np.float64)
    data["bs1_queries"] = q1
    data["bs1_eval"] = bs1.evaluate(q1)
    data["bs1_deriv_1"] = bs1.derivative(q1, [1])
    data["bs1_deriv_2"] = bs1.derivative(q1, [2])
    data["bs1_jacobian"] = bs1.jacobian(q1)
    data["bs1_basis"] = bs1.basis(q1)
    data["bs1_support"] = bs1.support(q1)

    # 2D Surface in 3D
    bs2 = splinepy.BSpline(
        degrees=[2, 3],
        knot_vectors=[
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0],
        ],
        control_points=RNG.uniform(-1.0, 2.0, size=(4 * 6, 3)),
    )
    q2 = make_queries(bs2.parametric_bounds)
    data["bs2_degrees"] = np.array(bs2.degrees, dtype=np.int32)
    data["bs2_kv_0"] = np.array(bs2.knot_vectors[0], dtype=np.float64)
    data["bs2_kv_1"] = np.array(bs2.knot_vectors[1], dtype=np.float64)
    data["bs2_cps"] = np.array(bs2.control_points, dtype=np.float64)
    data["bs2_queries"] = q2
    data["bs2_eval"] = bs2.evaluate(q2)
    data["bs2_deriv_10"] = bs2.derivative(q2, [1, 0])
    data["bs2_deriv_01"] = bs2.derivative(q2, [0, 1])
    data["bs2_deriv_11"] = bs2.derivative(q2, [1, 1])
    data["bs2_jacobian"] = bs2.jacobian(q2)
    data["bs2_basis"] = bs2.basis(q2)
    data["bs2_support"] = bs2.support(q2)

    # 3D Volume in 3D
    bs3 = splinepy.BSpline(
        degrees=[1, 2, 2],
        knot_vectors=[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        control_points=RNG.uniform(0.0, 10.0, size=(2 * 4 * 3, 3)),
    )
    q3 = make_queries(bs3.parametric_bounds)
    data["bs3_degrees"] = np.array(bs3.degrees, dtype=np.int32)
    data["bs3_kv_0"] = np.array(bs3.knot_vectors[0], dtype=np.float64)
    data["bs3_kv_1"] = np.array(bs3.knot_vectors[1], dtype=np.float64)
    data["bs3_kv_2"] = np.array(bs3.knot_vectors[2], dtype=np.float64)
    data["bs3_cps"] = np.array(bs3.control_points, dtype=np.float64)
    data["bs3_queries"] = q3
    data["bs3_eval"] = bs3.evaluate(q3)
    data["bs3_jacobian"] = bs3.jacobian(q3)
    data["bs3_basis"] = bs3.basis(q3)
    data["bs3_support"] = bs3.support(q3)

    np.savez(OUT_DIR / "eval_bspline.npz", **data)
    print("Saved eval_bspline.npz")

def generate_eval_nurbs():
    data = {}
    
    # 2D NURBS circle/disk in 2D
    r = 1.0
    w = 1.0 / np.sqrt(2.0)
    nurbs = splinepy.NURBS(
        degrees=[2, 1],
        knot_vectors=[
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
            [0, 0, 1, 1],
        ],
        control_points=[
            # inner radius 0.5
            [0.5, 0], [0.5, 0.5], [0, 0.5], [-0.5, 0.5], [-0.5, 0], [-0.5, -0.5], [0, -0.5], [0.5, -0.5], [0.5, 0],
            # outer radius 1.0
            [1.0, 0], [1.0, 1.0], [0, 1.0], [-1.0, 1.0], [-1.0, 0], [-1.0, -1.0], [0, -1.0], [1.0, -1.0], [1.0, 0],
        ],
        weights=[
            1, w, 1, w, 1, w, 1, w, 1,
            1, w, 1, w, 1, w, 1, w, 1,
        ],
    )
    qn = make_queries(nurbs.parametric_bounds, n=200)
    data["degrees"] = np.array(nurbs.degrees, dtype=np.int32)
    data["kv_0"] = np.array(nurbs.knot_vectors[0], dtype=np.float64)
    data["kv_1"] = np.array(nurbs.knot_vectors[1], dtype=np.float64)
    data["cps"] = np.array(nurbs.control_points, dtype=np.float64)
    data["weights"] = np.array(nurbs.weights, dtype=np.float64)
    data["queries"] = qn
    data["eval"] = nurbs.evaluate(qn)
    data["deriv_10"] = nurbs.derivative(qn, [1, 0])
    data["deriv_01"] = nurbs.derivative(qn, [0, 1])
    data["jacobian"] = nurbs.jacobian(qn)
    data["basis"] = nurbs.basis(qn)
    data["support"] = nurbs.support(qn)

    np.savez(OUT_DIR / "eval_nurbs.npz", **data)
    print("Saved eval_nurbs.npz")

def generate_eval_bezier():
    data = {}
    
    # 2D Bezier in 2D
    bez = splinepy.Bezier(
        degrees=[3, 2],
        control_points=RNG.uniform(0.0, 5.0, size=(4 * 3, 2)),
    )
    qb = make_queries(bez.parametric_bounds, n=150)
    data["degrees"] = np.array(bez.degrees, dtype=np.int32)
    data["cps"] = np.array(bez.control_points, dtype=np.float64)
    data["queries"] = qb
    data["eval"] = bez.evaluate(qb)
    data["deriv_10"] = bez.derivative(qb, [1, 0])
    data["deriv_01"] = bez.derivative(qb, [0, 1])
    data["deriv_20"] = bez.derivative(qb, [2, 0])
    data["jacobian"] = bez.jacobian(qb)
    data["basis"] = bez.basis(qb)
    data["support"] = bez.support(qb)

    # Rational Bezier
    rbez = splinepy.RationalBezier(
        degrees=[2, 2],
        control_points=RNG.uniform(0.0, 3.0, size=(3 * 3, 3)),
        weights=RNG.uniform(0.2, 2.0, size=9),
    )
    qrb = make_queries(rbez.parametric_bounds, n=150)
    data["r_degrees"] = np.array(rbez.degrees, dtype=np.int32)
    data["r_cps"] = np.array(rbez.control_points, dtype=np.float64)
    data["r_weights"] = np.array(rbez.weights, dtype=np.float64)
    data["r_queries"] = qrb
    data["r_eval"] = rbez.evaluate(qrb)
    data["r_deriv_10"] = rbez.derivative(qrb, [1, 0])
    data["r_jacobian"] = rbez.jacobian(qrb)

    np.savez(OUT_DIR / "eval_bezier.npz", **data)
    print("Saved eval_bezier.npz")

def generate_operations_bspline():
    data = {}
    
    bs = splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        control_points=RNG.uniform(0.0, 5.0, size=(3 * 3, 2)),
    )
    data["orig_degrees"] = np.array(bs.degrees, dtype=np.int32)
    data["orig_kv_0"] = np.array(bs.knot_vectors[0], dtype=np.float64)
    data["orig_kv_1"] = np.array(bs.knot_vectors[1], dtype=np.float64)
    data["orig_cps"] = np.array(bs.control_points, dtype=np.float64)

    # 1. Insert knots
    bs_ins = bs.copy()
    bs_ins.insert_knots(0, [0.3, 0.7])
    bs_ins.insert_knots(1, [0.5])
    data["ins_kv_0"] = np.array(bs_ins.knot_vectors[0], dtype=np.float64)
    data["ins_kv_1"] = np.array(bs_ins.knot_vectors[1], dtype=np.float64)
    data["ins_cps"] = np.array(bs_ins.control_points, dtype=np.float64)

    # 2. Elevate degrees
    bs_ele = bs.copy()
    bs_ele.elevate_degrees([0, 1])
    data["ele_degrees"] = np.array(bs_ele.degrees, dtype=np.int32)
    data["ele_kv_0"] = np.array(bs_ele.knot_vectors[0], dtype=np.float64)
    data["ele_kv_1"] = np.array(bs_ele.knot_vectors[1], dtype=np.float64)
    data["ele_cps"] = np.array(bs_ele.control_points, dtype=np.float64)

    # 3. Extract bezier patches
    patches = bs_ins.extract_bezier_patches()
    data["num_patches"] = len(patches)
    for i, p in enumerate(patches):
        data[f"patch_{i}_cps"] = np.array(p.control_points, dtype=np.float64)
        data[f"patch_{i}_degrees"] = np.array(p.degrees, dtype=np.int32)

    # 4. Extract boundaries
    for b_id in range(4):
        bnd = bs.extract.boundaries([b_id])[0]
        data[f"boundary_{b_id}_cps"] = np.array(bnd.control_points, dtype=np.float64)
        data[f"boundary_{b_id}_degrees"] = np.array(bnd.degrees, dtype=np.int32)

    # 5. Extract dim
    b_dim0 = splinepy.Bezier(degrees=bs.degrees, control_points=bs.control_points[:9]).extract_dim(0)
    data["b_dim0_cps"] = np.array(b_dim0.control_points, dtype=np.float64)

    np.savez(OUT_DIR / "operations_bspline.npz", **data)
    print("Saved operations_bspline.npz")

def generate_operations_bezier():
    data = {}
    
    b1 = splinepy.Bezier(
        degrees=[2, 2],
        control_points=RNG.uniform(0.0, 5.0, size=(9, 2)),
    )
    b2 = splinepy.Bezier(
        degrees=[2, 2],
        control_points=RNG.uniform(0.0, 2.0, size=(9, 2)),
    )
    data["b1_degrees"] = np.array(b1.degrees, dtype=np.int32)
    data["b1_cps"] = np.array(b1.control_points, dtype=np.float64)
    data["b2_degrees"] = np.array(b2.degrees, dtype=np.int32)
    data["b2_cps"] = np.array(b2.control_points, dtype=np.float64)

    # 1. Split
    splits = b1.split(0, [0.4])
    data["split0_cps"] = np.array(splits[0].control_points, dtype=np.float64)
    data["split1_cps"] = np.array(splits[1].control_points, dtype=np.float64)

    # 2. Add
    b_add = b1 + b2
    data["add_cps"] = np.array(b_add.control_points, dtype=np.float64)
    
    # 3. Multiply (scalar * 2D)
    scalar_b = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[[1.0], [2.0], [3.0], [4.0]],
    )
    data["scalar_b_degrees"] = np.array(scalar_b.degrees, dtype=np.int32)
    data["scalar_b_cps"] = np.array(scalar_b.control_points, dtype=np.float64)

    b_mul = b1 * scalar_b
    data["mul_cps"] = np.array(b_mul.control_points, dtype=np.float64)
    data["mul_degrees"] = np.array(b_mul.degrees, dtype=np.int32)

    # 4. Derivative spline
    b1_deriv_10 = b1.derivative_spline([1, 0])
    data["b1_deriv_10_degrees"] = np.array(b1_deriv_10.degrees, dtype=np.int32)
    data["b1_deriv_10_cps"] = np.array(b1_deriv_10.control_points, dtype=np.float64)

    b1_deriv_01 = b1.derivative_spline([0, 1])
    data["b1_deriv_01_degrees"] = np.array(b1_deriv_01.degrees, dtype=np.int32)
    data["b1_deriv_01_cps"] = np.array(b1_deriv_01.control_points, dtype=np.float64)

    # 5. Compose & sensitivities
    outer = splinepy.Bezier(degrees=[2, 2], control_points=RNG.uniform(0.0, 1.0, size=(9, 2)))
    inner = splinepy.Bezier(degrees=[1, 1], control_points=[[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])
    data["outer_degrees"] = np.array(outer.degrees, dtype=np.int32)
    data["outer_cps"] = np.array(outer.control_points, dtype=np.float64)
    data["inner_degrees"] = np.array(inner.degrees, dtype=np.int32)
    data["inner_cps"] = np.array(inner.control_points, dtype=np.float64)

    composed, sens = outer.compose(inner, compute_sensitivities=True)
    data["composed_cps"] = np.array(composed.control_points, dtype=np.float64)
    data["composed_degrees"] = np.array(composed.degrees, dtype=np.int32)
    data["num_sens"] = len(sens)
    for i, s in enumerate(sens):
        data[f"sens_{i}_cps"] = np.array(s.control_points, dtype=np.float64)

    np.savez(OUT_DIR / "operations_bezier.npz", **data)
    print("Saved operations_bezier.npz")

def generate_multipatch():
    data = {}
    
    # 2-patch multipatch (adjoining unit squares)
    p0 = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
        control_points=[[0, 0], [1, 0], [0, 1], [1, 1]],
    )
    p1 = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
        control_points=[[1, 0], [2, 0], [1, 1], [2, 1]],
    )
    mp = splinepy.Multipatch(splines=[p0, p1])
    mp.determine_interfaces()
    
    data["interfaces"] = np.array(mp.interfaces, dtype=np.int32)
    orientations = mp.interface_orientations()
    if orientations is not None and len(orientations) > 0:
        data["orientations"] = np.array(orientations, dtype=np.int32)

    bnd_mp = mp.boundary_multipatch()
    data["num_boundaries"] = len(bnd_mp.patches)
    for i, b in enumerate(bnd_mp.patches):
        data[f"bnd_{i}_cps"] = np.array(b.control_points, dtype=np.float64)

    np.savez(OUT_DIR / "multipatch_topology.npz", **data)
    print("Saved multipatch_topology.npz")

def export_binaries():
    bin_dir = OUT_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    for npz_path in OUT_DIR.glob("*.npz"):
        data = np.load(npz_path)
        prefix = npz_path.stem
        for k in data.files:
            arr = np.ascontiguousarray(data[k])
            out_file = bin_dir / f"{prefix}_{k}.bin"
            arr.tofile(out_file)
    print(f"Exported raw binaries to {bin_dir}/")

if __name__ == "__main__":
    if splinepy.__version__ != "0.2.1":
        raise SystemExit("Golden fixtures require the splinepy 0.2.1 oracle.")
    generate_eval_bspline()
    generate_eval_nurbs()
    generate_eval_bezier()
    generate_operations_bspline()
    generate_operations_bezier()
    generate_multipatch()
    export_binaries()
    print("All golden datasets generated successfully in tests/data/golden/!")
