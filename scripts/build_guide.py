"""Render docs/GUIDE.md from docs/GUIDE_TEMPLATE.md and outputs/summary.json.

Every number in the guide is a placeholder resolved here, so it is traceable to summary.json:
    {{partD/quantiles/2030_matures/P90|f1}}   value at that path, formatted
    {{derived/name|pct1}}                      a value derived below from summary.json only
    {{TABLE:name}}                             a table generated below from summary.json
A missing path is an error (no silent placeholders).

Formats: f0..f4 (fixed, thousands separator), pct0..pct3 (x100 with %), g2..g4 (significant digits),
sci0..sci2 (a×10ⁿ), int (rounded integer), raw.

Usage:  python scripts/build_guide.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt(v, spec: str) -> str:
    if spec == "raw" or isinstance(v, str):
        return str(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    v = float(v)
    if spec == "int":
        return f"{round(v):,}"
    return _fmt_num(v, spec).replace("-", "−")


def _fmt_num(v: float, spec: str) -> str:
    m = re.fullmatch(r"(f|pct|spct|g|sci)(\d)", spec)
    if not m:
        raise ValueError(f"unknown format {spec!r}")
    kind, n = m.group(1), int(m.group(2))
    if kind == "f":
        return f"{v:,.{n}f}"
    if kind == "pct":
        return f"{100 * v:.{n}f}%"
    if kind == "spct":
        return f"{100 * v:+.{n}f}%"
    if kind == "g":
        s = f"{v:.{n}g}"
        if "e" in s:
            return _fmt_num(v, f"sci{max(n - 1, 0)}")
        return s
    mant, exp = f"{v:.{n}e}".split("e")
    return f"{mant}×10{str(int(exp)).translate(SUP)}"


def get(ctx: dict, path: str):
    cur = ctx
    for part in path.split("/"):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"path {path!r}: {part!r} not found")
    return cur


# ---------------------------------------------------------------------------
# Derived values (computed from summary.json only)
# ---------------------------------------------------------------------------


def derived(s: dict) -> dict:
    A, B, C, D = s["partA"], s["partB"], s["partC"], s["partD"]
    so = np.array(C["sim_over_theory"])
    d = {
        "sim_sigma1_max_dev": float(np.max(np.abs(so[0] - 1))),
        "sim_sigma2_max_dev": float(np.max(np.abs(so[1] - 1))),
        "sim_sigma3_min_ratio": float(np.min(so[2])),
        "sim_sigma3_ratio_at_10": float(so[2][0]),
        "sim_sigma3_ratio_at_1e6": float(so[2][-1]),
        "p30_quantile_2030": D["pjm_30GW"]["2030_matures"]["quantile"],
        "pct_ratio_matures_P90_over_median_2030": D["quantiles"]["2030_matures"]["P90"] / D["quantiles"]["2030_matures"]["P50"],
        "mean_over_median_dq_2030": D["quantiles"]["2030_matures"]["mean"] / D["quantiles"]["2030_matures"]["P50"],
        "dq_at_prior_medians_2030": D["at_prior_medians"]["2030"]["PT_GW"] - D["at_prior_medians"]["2030"]["P0_GW"],
        "ladder_end_2030": D["fermi_ladder"]["2030_matures"][-1]["MW"],
        "ladder_start": D["fermi_ladder"]["2030_matures"][0]["MW"],
        "table4_n_rows": len(A["table4"]),
        "fw_P10_reldiff_2030": D["lognormal_check"]["2030_matures"]["fw"]["rel_diff"][0],
        "fw_P50_reldiff_2030": D["lognormal_check"]["2030_matures"]["fw"]["rel_diff"][1],
        "fw_P90_reldiff_2030": D["lognormal_check"]["2030_matures"]["fw"]["rel_diff"][2],
        "exact_max_abs_reldiff": max(abs(x) for k in D["lognormal_check"] for x in D["lognormal_check"][k]["exact"]["rel_diff"]),
        "fw_max_abs_reldiff": max(abs(x) for k in D["lognormal_check"] for x in D["lognormal_check"][k]["fw"]["rel_diff"]),
        "kappa_over_theory": B["meta"]["boundary"]["gamma"]["kappa_median"],
        "beta_mean": float(np.mean(list(A["summary"]["beta_by_case"].values()))),
        "half_gap_2030": D["halved_growth_mix_uncertainty"]["2030_matures"]["gap"],
        "ai_ratio_today_lo": C["ai_regime"]["ratio_day_ahead_vs_concurrency"][0],
        "ai_ratio_today_hi": C["ai_regime"]["ratio_day_ahead_vs_concurrency"][1],
        "ai_ratio_2030_lo": C["ai_regime"]["ratio_2030_vs_daily"][0],
        "ai_ratio_2030_hi": C["ai_regime"]["ratio_2030_vs_daily"][1],
        "ai_ratio_today_over_kappa": C["ai_regime"]["ratio_day_ahead_vs_concurrency"][0] / B["meta"]["boundary"]["gamma"]["kappa_median"],
        "max_aleatory_share_1e10": max(v for sg in C["aleatory_share_of_var_at_1e10_per_day"].values() for v in sg.values()),
        "rb_var_ratio": C["rao_blackwell_example"]["var_RB"] / C["rao_blackwell_example"]["var_S"],
        "rb_theory_ratio": C["rao_blackwell_example"]["theory_var_epistemic"] / C["rao_blackwell_example"]["theory_var"],
        "kr_gap_300": A["summary"]["kr_gap_uncertain"][2],
        "nv_gap_300_known": A["summary"]["nv_gap_known"][2],
        "nv_gap_375_known": A["summary"]["nv_gap_known"][0],
        "paper_gap_known_lo": min(A["summary"]["paper_gap_known"]),
        "paper_gap_known_hi": max(A["summary"]["paper_gap_known"]),
        "paper_gap_unc_hi": max(A["summary"]["paper_gap_uncertain"]),
        "sobol_max_diff": max(s["sobol"][k]["log_scale"]["max_abs_diff_vs_closed_form"] for k in s["sobol"]),
        "neg_share_2030_comp": D["quantiles"]["2030_compounding"]["p_negative"],
        "neg_share_2030_mat": D["quantiles"]["2030_matures"]["p_negative"],
        "cap_binding_frac": D["share_cap_binding_draws"] / D["n_draws"],
        "frac_disagree_theory_line": 1 - B["meta"]["boundary"]["gamma"]["frac_agree_theory_line"],
        "mix_shift_2030": D["fermi_ladder"]["2030_matures"][5]["factor"],
        "growth_2030": D["fermi_ladder"]["2030_matures"][4]["factor"],
        "eff_2030": D["fermi_ladder"]["2030_matures"][6]["factor"],
        "eff_2030_comp": D["fermi_ladder"]["2030_compounding"][6]["factor"],
        "ladder_end_2030_comp": D["fermi_ladder"]["2030_compounding"][-1]["MW"],
        "growth_plus_mix": s["sobol"]["matures"]["log_scale"]["S1"]["growth"] + s["sobol"]["matures"]["log_scale"]["S1"]["mix"],
        "agentic_term_log_sd": float(np.hypot(*[p["log_sd"] for p in D["priors"] if p["name"] in ("a", "w_AT")])),
    }
    return d


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def se(x: float) -> str:
    """Standard error to one significant digit (e.g. 0.004)."""
    return f"{x:.1g}"


def nice(x: float) -> str:
    """8e9 -> 8×10⁹; ordinary numbers unchanged."""
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        mant, exp = f"{x:.2g}".replace("e+", "e").split("e")
        return f"{mant}×10{str(int(exp)).translate(SUP)}"
    return f"{x:g}"


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    header = [h.replace("|", "\\|") for h in header]
    rows = [[re.sub(r"(?<![\w\-/.])-(?=\d)", "−", c).replace("|", "\\|") for c in r] for r in rows]
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def t_repro23(s):
    rows = []
    for r in s["partA"]["table2_3"]:
        rows.append([r["case"], f"{r['b_star']} ({r['paper_b_star']})", f"{r['Pi_star']:.2f} ({r['paper_Pi_star']:.2f})",
                     f"{r['b_bar']} ({r['paper_b_bar']})", f"{r['Pi_bar']:.2f} ({r['paper_Pi_bar']:.2f})",
                     f"{100 * r['gap_bar']:.2f}% ({100 * r['paper_gap_bar']:.3g}%)", str(r["b_det"]),
                     f"{100 * r['gap_det']:.2f}%", f"{r['beta_det']:.2f}"])
    return md_table(["case", "b* (paper)", "Π* (paper)", "b̄ (paper)", "Π(b̄) (paper)", "gap(b̄) (paper)", "b_det",
                     "gap(b_det)", "β of b_det"], rows)


def t_repro_ext(s, key):
    rows = []
    for r in s["partA"][key]:
        ok = "✓" if r["match_b"] else "✗"
        rows.append([r["case"], f"{100 * r['cv']:.1f}%", f"{r['b_star']} / {r['paper_b_star']}",
                     f"{r['Pi_star']:.2f} / {r['paper_Pi_star']:.2f}", f"{r['b_bar']} / {r['paper_b_bar']}",
                     f"{r['Pi_bar']:.2f} / {r['paper_Pi_bar']:.2f}", f"{100 * r['gap_bar']:.2f}% / {100 * r['paper_gap_bar']:.3g}%",
                     ok, f"{r['max_abs_dPi']:.3f}"])
    return md_table(["case", "CV", "b* ours / paper", "Π* ours / paper", "b̄ ours / paper", "Π(b̄) ours / paper",
                     "gap ours / paper", "b match", "max |ΔΠ|"], rows)


def t_quantiles(s):
    D = s["partD"]
    rows = []
    for key, label in [("2028_matures", "2028/29 · matures"), ("2028_compounding", "2028/29 · compounding"),
                       ("2030_matures", "2030/31 · matures"), ("2030_compounding", "2030/31 · compounding")]:
        q = D["quantiles"][key]
        rows.append([label] + [f"{q[p]:.2f} ± {se(q[p + '_se'])}" for p in ("P10", "P50", "P90", "P95")]
                    + [f"{q['mean']:.2f} ± {se(q['mean_se'])}", f"{q['p_exceed']:.3f} ± {se(q['p_exceed_se'])}",
                       f"{100 * q['p_negative']:.1f}%"])
    return md_table(["delivery year · efficiency", "P10", "P50", "P90", "P95", "mean", "P(ΔQ > 30 GW)", "P(ΔQ < 0)"], rows)


def t_capacity(s):
    D = s["partD"]
    rows = []
    for r in D["cost_ratios"]:
        k = f"{r:.2f}"
        rows.append([k, f"P{round(100 * (1 - r))}"] + [f"{D['quantile_capacity'][c][k]:.1f}" for c in
                                                       ("2028_matures", "2028_compounding", "2030_matures", "2030_compounding")])
    return md_table(["c_cap/c_short", "quantile", "2028/29 matures", "2028/29 compounding", "2030/31 matures",
                     "2030/31 compounding"], rows)


def t_ladder(s):
    L, Lc = s["partD"]["fermi_ladder"]["2030_matures"], s["partD"]["fermi_ladder"]["2030_compounding"]
    rows = []
    for a, b in zip(L, Lc):
        fa = "start" if a["factor"] is None else f"×{a['factor']:.3g}"
        fb = "" if a["factor"] == b["factor"] else f" (compounding: ×{b['factor']:.3g} → {b['MW']:,.0f} MW)"
        rows.append([a["step"], fa, f"{a['MW']:,.0f}" + fb])
    return md_table(["step (every parameter at its prior median; DY 2030/31)", "factor", "MW"], rows, "lrl")


def t_lognormal(s):
    rows = []
    for key in ("2028_matures", "2028_compounding", "2030_matures", "2030_compounding"):
        c = s["partD"]["lognormal_check"][key]
        rows.append([key.replace("_", " · "), " / ".join(f"{x:.2f}" for x in c["mc"]),
                     " / ".join(f"{100 * x:+.0f}%" for x in c["fw"]["rel_diff"]),
                     " / ".join(f"{100 * x:+.1f}%" for x in c["exact"]["rel_diff"]),
                     f"{c['mc_var_log']:.3f} / {c['fw']['var_log']:.3f} / {c['exact']['var_log']:.3f}"])
    c = s["partD"]["P0_lognormal_check"]
    rows.append(["P₀ (2025)", " / ".join(f"{x:.2f}" for x in c["mc"]), " / ".join(f"{100 * x:+.0f}%" for x in c["fw"]["rel_diff"]),
                 " / ".join(f"{100 * x:+.1f}%" for x in c["exact"]["rel_diff"]),
                 f"{c['mc_var_log']:.3f} / {c['fw']['var_log']:.3f} / {c['exact']['var_log']:.3f}"])
    return md_table(["P_T (GW)", "MC P10 / P50 / P90", "FW error", "exact-log-moment error", "Var ln P: MC / FW / exact"], rows)


def t_sobol(s):
    rows = []
    groups = s["sobol"]["matures"]["groups"]
    for g in groups:
        r = [g]
        for k in ("matures", "compounding"):
            L = s["sobol"][k]["log_scale"]
            r += [f"{L['S1'][g]:.3f} ± {L['S1_conf'][g]:.3f}", f"{L['closed_form_S1'][g]:.3f}", f"{L['ST'][g]:.3f}"]
        rows.append(r)
    return md_table(["group", "S₁ SALib (matures)", "S₁ closed form", "S_T", "S₁ SALib (compounding)",
                     "S₁ closed form", "S_T"], rows)


def t_sobol_lin(s):
    rows = []
    L = s["sobol"]["matures"]["linear_scale"]
    for g in s["sobol"]["matures"]["groups"]:
        rows.append([g, f"{L['S1'][g]:.3f} ± {L['S1_conf'][g]:.3f}", f"{L['lognormal_formula_S1'][g]:.3f}"])
    return md_table(["group", "S₁ on P_T itself (SALib)", "(e^{s²ᵢ} − 1)/(e^{S²} − 1)"], rows)


def t_priors(s):
    rows = []
    for p in s["partD"]["priors"]:
        rng = f"{nice(p['lo'])} – {nice(p['hi'])}"
        kind = "lognormal (P5–P95)" if p["dist"] == "lognormal" else "uniform"
        rows.append([p["symbol"], p["description"], kind, rng, nice(float(f"{p['median']:.3g}")), p["group"],
                     p["source"] + (" *(assumption)*" if p["assumption"] else "")])
    return md_table(["symbol", "parameter", "distribution", "range", "median", "group", "source"], rows, "lllrrll")


def t_cox(s):
    C = s["partC"]
    rows = []
    for j, n in enumerate(C["nbar"]):
        rows.append([f"{n:,.0f}", str(C["n_days"][0][j])] + [f"{C['cv_sim'][i][j]:.3f} / {C['cv_theory'][i][j]:.3f}"
                                                              for i in range(3)])
    return md_table(["n̄ (queries/day)", "days simulated", "σ = 1: sim / theory", "σ = 2: sim / theory",
                     "σ = 3: sim / theory"], rows)


def t_kappa(s):
    B = s["partB"]["meta"]
    rows = []
    for E, kg, kl in zip(B["E_values"], B["boundary"]["gamma"]["kappa_by_E"], B["boundary"]["lognormal"]["kappa_by_E"]):
        cell = lambda k: "no switch on the CV grid" if k is None or not np.isfinite(k) else f"{k:.2f}"  # noqa: E731
        rows.append([f"{E:,.0f}", cell(kg), cell(kl)])
    return md_table(["offered load E", "κ = CV·√E at the switch (gamma Λ)", "κ (lognormal Λ)"], rows)


def t_halved(s):
    D = s["partD"]["halved_growth_mix_uncertainty"]
    rows = []
    for key in ("2028_matures", "2028_compounding", "2030_matures", "2030_compounding"):
        h = D[key]
        rows.append([key.replace("_", " · "), f"{h['base_gap']:.1f}", f"{h['gap']:.1f}", f"{100 * h['shrink_fraction']:.0f}%",
                     f"{s['partD']['quantiles'][key]['p_exceed']:.3f} → {h['p_exceed_30']:.3f}"])
    return md_table(["case", "P90 − median (GW), base", "with half the growth & mix uncertainty", "reduction",
                     "P(ΔQ > 30 GW)"], rows)


TABLES = {"repro23": t_repro23, "repro4": lambda s: t_repro_ext(s, "table4"), "repro7": lambda s: t_repro_ext(s, "table7"),
          "quantiles": t_quantiles, "capacity": t_capacity, "ladder": t_ladder, "lognormal": t_lognormal,
          "sobol": t_sobol, "sobol_lin": t_sobol_lin, "priors": t_priors, "cox": t_cox, "kappa": t_kappa,
          "halved": t_halved}


def render(template: str, s: dict) -> str:
    ctx = dict(s)
    ctx["derived"] = derived(s)

    def table(m):
        return TABLES[m.group(1)](s)

    def value(m):
        path, spec = m.group(1), (m.group(2) or "g3")
        return fmt(get(ctx, path), spec)

    out = re.sub(r"\{\{TABLE:(\w+)\}\}", table, template)
    out = re.sub(r"\{\{([^}|]+?)(?:\|(\w+))?\}\}", value, out)
    left = re.findall(r"\{\{.*?\}\}", out)
    if left:
        raise ValueError(f"unresolved placeholders: {left[:5]}")
    return out


def main() -> None:
    s = json.loads((ROOT / "outputs" / "summary.json").read_text())
    tpl = (ROOT / "docs" / "GUIDE_TEMPLATE.md").read_text()
    out = render(tpl, s)
    (ROOT / "docs" / "GUIDE.md").write_text(out)
    n = len(re.findall(r"\{\{", tpl))
    print(f"wrote docs/GUIDE.md ({len(out.splitlines())} lines, {n} placeholders resolved)")


if __name__ == "__main__":
    sys.exit(main())
