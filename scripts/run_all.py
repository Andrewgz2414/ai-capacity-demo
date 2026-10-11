"""Run every part of the demo and write outputs/summary.json, the four figures and the handoff.

Usage:
    python scripts/run_all.py                # uses the caches in outputs/cache if present
    python scripts/run_all.py --recompute    # rebuilds the regime map and Cox caches (~2 min)
    python scripts/run_all.py --no-sobol     # skip the optional sensitivity analysis

Every number quoted in docs/GUIDE.md is read from outputs/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from aicap import OUTPUTS, cox, plotting, run_info  # noqa: E402
from aicap import pjm_load as pl  # noqa: E402
from aicap.capacity_rules import (PAPER_TABLE2_3, PAPER_TABLE4, PAPER_TABLE7, fit_safety_factor,  # noqa: E402
                                  load_regime_grid, min_second_difference, regime_cache_path,
                                  reproduction_rows)
from aicap.erlang_a import SECTION4, CostFunction, UniformRate, expected_cost  # noqa: E402
from aicap.params import (COST_RATIOS, EFFICIENCY_STRUCTURES, N_DRAWS, PJM_DC_GROWTH_GW,  # noqa: E402
                          PJM_SUMMER_PEAK_2026_GW, PRIORS, SEED)

COX_CACHE = OUTPUTS / "cache" / "cox_sim.npz"
COX_SETTINGS = dict(nbar=np.logspace(1, 6, 11), sigmas=(1.0, 2.0, 3.0), cv_rate=0.5, seed=20251011, budget=2e8)


def jsonable(x):
    """Recursively convert numpy types for json.dump."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


def part_a() -> dict:
    t = time.time()
    rows23 = reproduction_rows(PAPER_TABLE2_3, source="Tables 2-3")
    rows4 = reproduction_rows(PAPER_TABLE4, source="Table 4")
    rows7 = reproduction_rows(PAPER_TABLE7, source="Table 7")
    fixed = [r for r in rows23 if r["cv"] == 0]
    beta_fit = fit_safety_factor([r["b_det"] for r in fixed], [r["E"] for r in fixed])
    # quadrature convergence (Gauss-Legendre, 64 vs 128 nodes) and discrete convexity checks
    gl, convex = [], []
    for a in (25, 50, 200):
        rate = UniformRate(a, 2 * a)
        for b in range(a, 2 * a + 1, max(1, a // 5)):
            p64 = expected_cost(b, rate, method="gauss", n_nodes=64)
            p128 = expected_cost(b, rate, method="gauss", n_nodes=128)
            gl.append(abs(p64 / p128 - 1))
        cost = CostFunction(rate, SECTION4)
        convex.append(min_second_difference([cost(b) for b in range(0, 3 * a)]))
    unc = [r for r in rows23 if r["cv"] > 0]
    summary = {
        "nv_gap_uncertain": [r["gap_bar"] for r in unc],
        "nv_gap_known": [r["gap_bar"] for r in fixed],
        "kr_gap_uncertain": [r["gap_det"] for r in unc],
        "paper_gap_uncertain": [r["paper_gap_bar"] for r in unc],
        "paper_gap_known": [r["paper_gap_bar"] for r in fixed],
        "nv_gap_uncertain_max": max(r["gap_bar"] for r in unc),
        "nv_gap_known_min": min(r["gap_bar"] for r in fixed),
        "nv_gap_known_max": max(r["gap_bar"] for r in fixed),
        "table4_all_b_match": all(r["match_b"] for r in rows4),
        "table4_max_abs_dPi": max(r["max_abs_dPi"] for r in rows4),
        "table23_b_star_minus_paper": [r["b_star"] - r["paper_b_star"] for r in rows23],
        "table23_max_abs_dPi": max(r["max_abs_dPi"] for r in rows23),
        "table7_max_abs_dPi": max(r["max_abs_dPi"] for r in rows7),
        "beta_fit_fixed_cases": beta_fit,
        "beta_by_case": {r["case"]: r["beta_det"] for r in fixed},
        "gl_convergence_max_rel_diff_64_vs_128": max(gl),
        "min_second_difference": min(convex),
    }
    fig = plotting.fig1_reproduction(rows23)
    plotting.save(fig, "fig1_reproduction")
    print(f"[A] reproduction done in {time.time() - t:.1f} s; Table 4 all b match: {summary['table4_all_b_match']}")
    return {"params": SECTION4.__dict__ | {"fractile": SECTION4.fractile, "queue_weight": SECTION4.queue_weight},
            "table2_3": rows23, "table4": rows4, "table7": rows7, "summary": summary}


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def part_b(recompute: bool) -> dict:
    path = regime_cache_path()
    if recompute or not path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("precompute_regime", ROOT / "scripts" / "precompute_regime.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main(path)
    g = load_regime_grid(path)
    meta = g["meta"]
    gam = {"E": g["E"], "cv": g["cv"], "log_ratio": g["gamma_log_ratio"]}
    # a few cells for the guide: the paper's three CV = 19% cases are not on the grid, so report grid corners
    corners = {}
    for (i, j), name in {(0, 0): "E10_CV0.01", (0, -1): "E10_CV0.6", (-1, 0): "E3000_CV0.01",
                         (-1, -1): "E3000_CV0.6"}.items():
        corners[name] = {k: float(g[f"gamma_{k}"][i, j]) for k in ("gap_bar", "gap_det", "b_star", "b_bar", "b_det")}
    print(f"[B] regime cache loaded ({path.name}); kappa = {meta['boundary']['gamma']['kappa_median']:.2f}")
    return {"meta": meta, "grid_corners": corners, "frac_cells_nv_better": float(np.mean(g["gamma_log_ratio"] < 0)),
            "_grid": gam}


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------


def cox_simulation(recompute: bool) -> dict:
    if COX_CACHE.exists() and not recompute:
        with np.load(COX_CACHE) as z:
            return {k: z[k] for k in z.files}
    t = time.time()
    s = COX_SETTINGS
    sim = cox.simulate_cv_curve(s["nbar"], s["sigmas"], s["cv_rate"], seed=s["seed"], budget=s["budget"])
    COX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COX_CACHE, **{k: np.asarray(v) for k, v in sim.items()})
    print(f"[C] Cox simulation rebuilt in {time.time() - t:.1f} s")
    with np.load(COX_CACHE) as z:
        return {k: z[k] for k in z.files}


def part_c(recompute: bool) -> dict:
    sim = cox_simulation(recompute)
    cvr = float(sim["cv_rate"])
    theory = np.array([cox.cv_daily_energy(sim["nbar"], s, cvr) for s in sim["sigmas"]])
    rb = cox.rao_blackwell_demo(1000, 2.0, cvr, 20_000, np.random.default_rng(SEED))
    aleatory_share_grid = {f"sigma{s:g}": {f"cv{c:g}": float((np.exp(s**2) / 1e10) / (np.exp(s**2) / 1e10 + c**2))
                                           for c in (0.05, 0.2, 0.5)} for s in (1.0, 2.0, 3.0)}
    out = {
        "cv_rate": cvr, "sigmas": sim["sigmas"], "nbar": sim["nbar"], "n_days": sim["n_days"],
        "cv_sim": sim["cv_hat"], "cv_sim_lo": sim["cv_lo"], "cv_sim_hi": sim["cv_hi"], "cv_theory": theory,
        "sim_over_theory": sim["cv_hat"] / theory, "seed": int(sim["seed"]), "budget": float(sim["budget"]),
        "crossover_cv_rate_0.5": {f"sigma{s:g}": cox.crossover_n(s, 0.5) for s in (1.0, 2.0, 3.0)},
        "crossover_cv_rate_1": {f"sigma{s:g}": cox.crossover_n(s, 1.0) for s in (1.0, 2.0, 3.0)},
        "mean_over_median": {f"sigma{s:g}": cox.mean_over_median(s) for s in (0.7, 1.0, 1.3, 2.0, 3.0)},
        "energy_share_top10pct": {f"sigma{s:g}": cox.energy_share_above_quantile(s, 0.9) for s in (1.0, 2.0, 3.0)},
        "energy_share_top1pct": {f"sigma{s:g}": cox.energy_share_above_quantile(s, 0.99) for s in (1.0, 2.0, 3.0)},
        "rao_blackwell_example": {"nbar": 1000, "sigma": 2.0, "cv_rate": cvr, "days": 20_000,
                                  "mean_S": rb["mean_S"], "mean_RB": rb["mean_RB"], "var_S": rb["var_S"],
                                  "var_RB": rb["var_RB"], "theory_var": rb["theory"]["var"],
                                  "theory_var_epistemic": rb["theory"]["var_epistemic"]},
        "aleatory_share_of_var_at_1e10_per_day": aleatory_share_grid,
    }
    fig = plotting.fig3_cox_floor({"nbar": sim["nbar"], "sigmas": sim["sigmas"], "cv_hat": sim["cv_hat"]}, cvr,
                                  cox.cv_daily_energy, cox.crossover_n)
    plotting.save(fig, "fig3_cox_floor")
    print("[C] Cox figure saved")
    return out


# ---------------------------------------------------------------------------
# Part D
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optional: variance-based sensitivity (Sobol) on ln P_T
# ---------------------------------------------------------------------------


def sobol_part(T: float = 5, structure: str = "matures", N: int = 2**14) -> dict:
    from scipy import integrate
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    names = list(PRIORS)
    groups = [PRIORS[n].group for n in names]
    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]] * len(names), "groups": groups}
    X = sobol_sample.sample(problem, N, calc_second_order=False, seed=SEED)
    th = {n: PRIORS[n].ppf(np.clip(X[:, i], 1e-12, 1 - 1e-12)) for i, n in enumerate(names)}
    PT = pl.pT_gw(th, T, structure)
    res_log = sobol_analyze.analyze(problem, np.log(PT), calc_second_order=False, seed=SEED)
    res_lin = sobol_analyze.analyze(problem, PT, calc_second_order=False, seed=SEED)
    gnames = list(dict.fromkeys(groups))

    def by_name(res):
        total, first = res.to_df()
        return ({g: float(first.loc[g, "S1"]) for g in gnames}, {g: float(first.loc[g, "S1_conf"]) for g in gnames},
                {g: float(total.loc[g, "ST"]) for g in gnames}, {g: float(total.loc[g, "ST_conf"]) for g in gnames})
    S1, S1c, ST, STc = by_name(res_log)
    L1, L1c, _, _ = by_name(res_lin)
    # closed-form first-order indices on the log scale
    lm = pl.log_moments(T, structure, mix_method="exact")
    parts = lm["parts"]
    total = lm["var_log"]
    lam_lo, lam_hi = PRIORS["lam"].lo, PRIORS["lam"].hi
    phi = lambda l: (1 - np.exp(-l * T)) / l  # noqa: E731
    Ephi = integrate.quad(phi, lam_lo, lam_hi)[0] / (lam_hi - lam_lo)
    Ephi2 = integrate.quad(lambda l: phi(l) ** 2, lam_lo, lam_hi)[0] / (lam_hi - lam_lo)
    mg, vg, mh, vh = PRIORS["g1"].mu, PRIORS["g1"].s ** 2, PRIORS["h"].mu, PRIORS["h"].s ** 2
    if structure == "matures":
        v_growth = (vg + (mg - mh) ** 2) * Ephi2 - (mg - mh) ** 2 * Ephi**2  # Var E[ln P | g1, lam]
        v_eff = vh * Ephi**2  # Var E[ln P | h]
    else:
        v_growth = (vg + mg**2) * Ephi2 - mg**2 * Ephi**2
        v_eff = T**2 * vh
    closed = {"growth": v_growth / total, "efficiency": v_eff / total, "mix": parts["mix"][1] / total,
              "base volume": parts["base volume"][1] / total, "Wh per query": parts["Wh per query"][1] / total,
              "training": parts["training"][1] / total, "siting and peak": parts["siting and peak"][1] / total}
    # linear scale: S_i = (e^{s_i^2} - 1)/(e^{S^2} - 1) if each group factor were lognormal
    s2 = {"mix": parts["mix"][1], "base volume": parts["base volume"][1], "Wh per query": parts["Wh per query"][1],
          "training": parts["training"][1], "siting and peak": parts["siting and peak"][1],
          "growth": v_growth, "efficiency": v_eff}
    S2 = total
    closed_lin = {k: float(np.expm1(v) / np.expm1(S2)) for k, v in s2.items()}
    out = {"T": T, "structure": structure, "N_base": N, "n_model_runs": int(X.shape[0]),
           "groups": gnames,
           "log_scale": {"S1": S1, "S1_conf": S1c, "ST": ST, "ST_conf": STc,
                         "closed_form_S1": closed, "sum_closed_form": float(sum(closed.values())),
                         "max_abs_diff_vs_closed_form": float(max(abs(S1[g] - closed[g]) for g in gnames))},
           "linear_scale": {"S1": L1, "S1_conf": L1c, "lognormal_formula_S1": closed_lin}}
    print("[Sobol] done")
    return out


# ---------------------------------------------------------------------------


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rebuild the regime and Cox caches")
    ap.add_argument("--no-sobol", action="store_true", help="skip the optional Sobol analysis")
    args = ap.parse_args(argv)
    t0 = time.time()
    info = run_info()
    summary = {"run": info}
    summary["partA"] = part_a()
    summary["partB"] = part_b(args.recompute)
    summary["partC"] = part_c(args.recompute)
    sim = pl.simulate(n=N_DRAWS, seed=SEED)
    summary["partD"] = pl.summarize(sim)
    # AI regime numbers need the delivery-year CV from part D
    ai = cox.ai_regime_numbers(cv_2030=pl.cv_2030_range(sim))
    summary["partC"]["ai_regime"] = ai
    # Figure 2 (needs B, C, D)
    kappa = summary["partB"]["meta"]["boundary"]["gamma"]["kappa_median"]
    boxes = cox.regime_boxes(ai)
    fig = plotting.fig2_regime_map(summary["partB"].pop("_grid"), kappa, boxes)
    plotting.save(fig, "fig2_regime_map")
    # Figure 4
    fig = plotting.fig4_handoff(sim[(2030, "matures")]["dQ"], sim[(2030, "compounding")]["dQ"], COST_RATIOS,
                                pjm_level=PJM_DC_GROWTH_GW, pjm_peak=PJM_SUMMER_PEAK_2026_GW)
    plotting.save(fig, "fig4_handoff")
    pl.write_handoff(sim, summary["partD"], info)
    if not args.no_sobol:
        summary["sobol"] = {f"{s}": sobol_part(5, s) for s in EFFICIENCY_STRUCTURES}
    summary["runtime_s"] = time.time() - t0
    out = OUTPUTS / "summary.json"
    out.write_text(json.dumps(jsonable(summary), indent=1))
    print(f"wrote {out} ({time.time() - t0:.1f} s total)")
    return summary


if __name__ == "__main__":
    main()
