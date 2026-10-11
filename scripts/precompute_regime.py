"""Build the regime map cache outputs/cache/regime_grid.npz (spec section 4).

Grid: offered load E log-spaced 10..3000 (15 values) x rate CV log-spaced 0.01..0.6 (15 values),
section 4 cost parameters, gamma-distributed Lambda (main) and lognormal (robustness, in the
spirit of the paper's section 6). Pi is integrated with the "split" method (closed-form
newsvendor part + local composite Gauss-Legendre); a refined rerun (16 nodes per panel, half
the panel width) checks quadrature convergence.

Usage:  python scripts/precompute_regime.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aicap import run_info  # noqa: E402
from aicap.capacity_rules import GAP_FLOOR, boundary_analysis, regime_cache_path, regime_grid  # noqa: E402
from aicap.erlang_a import SECTION4  # noqa: E402

E_VALUES = np.geomspace(10, 3000, 15)
CV_VALUES = np.geomspace(0.01, 0.6, 15)
FIELDS = ["b_star", "Pi_star", "b_bar", "Pi_bar", "gap_bar", "b_det", "Pi_det", "gap_det", "log_ratio", "n_evals"]


def main(path=None) -> dict:
    path = Path(path) if path else regime_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    grids, timings = {}, {}
    for dist in ("gamma", "lognormal"):
        t = time.time()
        grids[dist] = regime_grid(E_VALUES, CV_VALUES, dist)
        timings[dist] = time.time() - t
        print(f"{dist:9s} grid: {timings[dist]:.1f} s")
    t = time.time()
    refined = regime_grid(E_VALUES, CV_VALUES, "gamma", nodes_per_panel=16, width_factor=0.5)
    timings["gamma_refined"] = time.time() - t
    g = grids["gamma"]
    conv = {
        "max_rel_diff_Pi_star": float(np.max(np.abs(refined["Pi_star"] / g["Pi_star"] - 1))),
        "max_rel_diff_Pi_bar": float(np.max(np.abs(refined["Pi_bar"] / g["Pi_bar"] - 1))),
        "max_rel_diff_Pi_det": float(np.max(np.abs(refined["Pi_det"] / g["Pi_det"] - 1))),
        "max_abs_diff_gap_bar": float(np.max(np.abs(refined["gap_bar"] - g["gap_bar"]))),
        "max_abs_diff_gap_det": float(np.max(np.abs(refined["gap_det"] - g["gap_det"]))),
        "same_b_star_all_cells": bool(np.array_equal(refined["b_star"], g["b_star"])),
    }
    print("convergence (refined vs default quadrature):", conv)
    bounds = {d: boundary_analysis(grids[d]) for d in grids}
    ln = grids["lognormal"]
    robust = {
        "frac_same_cheaper_rule": float(np.mean((g["log_ratio"] < 0) == (ln["log_ratio"] < 0))),
        "max_abs_diff_log_ratio": float(np.max(np.abs(g["log_ratio"] - ln["log_ratio"]))),
        "max_abs_diff_gap_bar": float(np.max(np.abs(g["gap_bar"] - ln["gap_bar"]))),
        "max_abs_diff_gap_det": float(np.max(np.abs(g["gap_det"] - ln["gap_det"]))),
    }
    print("boundary:", {d: {k: v for k, v in b.items() if k != "kappa_by_E"} for d, b in bounds.items()})
    print("lognormal robustness:", robust)
    meta = {
        "run": run_info(), "params": SECTION4.__dict__, "gap_floor": GAP_FLOOR,
        "E_values": E_VALUES.tolist(), "cv_values": CV_VALUES.tolist(),
        "integration": "split: closed-form newsvendor + composite Gauss-Legendre (8 nodes/panel) "
                       "on |lam - b theta| <~ 12 sqrt(b); kink at lam = b theta is a panel edge",
        "convergence": conv,
        "boundary": {d: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
                     for d, b in bounds.items()},
        "lognormal_robustness": robust, "timings_s": timings,
    }
    arrays = {"E": E_VALUES, "cv": CV_VALUES}
    for d, grid in grids.items():
        for f in FIELDS:
            arrays[f"{d}_{f}"] = grid[f]
    np.savez_compressed(path, meta_json=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {path}  (total {time.time() - t0:.1f} s)")
    return meta


if __name__ == "__main__":
    main()
