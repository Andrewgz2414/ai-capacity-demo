"""Regenerate notebooks/demo.ipynb from this script (usage: python scripts/build_notebook.py notebooks/demo.ipynb).

The notebook is the deliverable; this builder keeps its cells reviewable as plain Python.
"""
import sys
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# What does AI really cost? From one query to PJM's capacity market

**Presenter 2: code demo (≈12 min).** Presenter 1's Wh per query → a distribution of new PJM summer-peak load ΔQ (GW) → why the capacity to buy is a *quantile* of ΔQ, not its median → samples handed to presenter 3.

Primary reference: Bassamboo, Randhawa & Zeevi (2010), *Capacity sizing under parameter uncertainty: safety staffing principles revisited*, Management Science 56(10).
""")

md("""
### 0 · Setup *(run before the talk)*
""")
code(r'''
%matplotlib inline
import sys
from pathlib import Path
try:
    import aicap
except ImportError:  # not pip-installed: use the repository's src/ folder
    sys.path.insert(0, str(Path.cwd().parent / "src"))
    import aicap

import numpy as np
import pandas as pd
from IPython.display import display

from aicap import cox, plotting, run_info
from aicap import pjm_load as pl
from aicap.capacity_rules import PAPER_TABLE2_3, load_regime_grid, regime_grid, boundary_analysis, reproduction_rows
from aicap.params import (COST_RATIOS, DELIVERY_YEARS, PJM_DC_GROWTH_GW, PJM_SUMMER_PEAK_2026_GW, PRESENTER1_STATUS,
                          PRIORS, SEED, N_DRAWS)

RECOMPUTE = False   # True rebuilds the regime map (~1 min) and the Cox simulation (~45 s) instead of loading caches
plotting.apply_style()
pd.set_option("display.precision", 3)

def show(df, fmt=None):
    """Display a table in a projector-friendly font size (numbers to 3 significant digits by default)."""
    default = lambda v: f"{v:.3g}" if isinstance(v, (int, float, np.number)) else v
    sty = df.style.format(default, na_rep="")
    if fmt:
        sty = sty.format(fmt, na_rep="")
    sty = sty.set_table_styles(
        [{"selector": "td, th", "props": [("font-size", "15px"), ("padding", "4px 10px")]}])
    display(sty)

sim = pl.simulate(n=N_DRAWS, seed=SEED)   # 200,000 parameter draws (0.05 s); used in cells 3, 5 and 6
print(run_info())
''')

md("""
### 1 · From presenter 1: energy per query *(1 min)*

Per-query energy is lognormal with median $M$ and log-sd $\\sigma$. Summed over billions of queries only the **mean** survives:
$$E[X] = M\\,e^{\\sigma^2/2}$$
""")
code(r'''
Mc, sc = PRIORS["M_c"].median, PRIORS["sigma_c"].median
types = {"chat": 1.0, "reasoning": PRIORS["r"].median, "agentic (per task)": PRIORS["a"].median}
tab = pd.DataFrame({k: {"median Wh": Mc * m, "σ": sc, "mean Wh = median·e^(σ²/2)": Mc * m * np.exp(sc**2 / 2),
                        "energy share of top 10% of queries": cox.energy_share_above_quantile(sc, 0.9)}
                    for k, m in types.items()}).T
show(tab, {"median Wh": "{:.2f}", "σ": "{:.2f}", "mean Wh = median·e^(σ²/2)": "{:.2f}",
           "energy share of top 10% of queries": "{:.0%}"})
print(f"mean/median = e^(σ²/2) = {cox.mean_over_median(sc):.2f}   ({PRESENTER1_STATUS})")
''')

md("""
### 2 · The paper: capacity under an uncertain arrival rate *(3 min)*

Erlang-A queue (M/M/b+M): Poisson arrivals at rate $\\Lambda$ (itself random), $b$ servers, service rate $\\mu=1$, abandonment $\\gamma=3$. Capacity is fixed **before** $\\Lambda$ is known:
$$\\Pi(b) = c\\,b + (h + p\\gamma)\\,E\\big[(N-b)^+\\big]$$
Newsvendor rule (ignore Poisson noise, hedge only the rate): $P(\\Lambda > \\mu\\bar b) = \\dfrac{c}{\\mu(p + h/\\gamma)} = \\tfrac14$, so $\\bar b$ is the 75th percentile of $\\Lambda$.
Known-rate rule: the best $b$ if $\\Lambda$ were exactly its mean ($\\approx E + \\beta\\sqrt{E}$).
""")
code(r'''
rows = reproduction_rows(PAPER_TABLE2_3)
rep = pd.DataFrame(rows).set_index("case")[["b_star", "Pi_star", "b_bar", "Pi_bar", "gap_bar", "paper_gap_bar",
                                            "b_det", "gap_det", "beta_det"]]
rep.columns = ["b*", "Π*", "b̄", "Π(b̄)", "gap(b̄)", "gap(b̄) paper", "b_det", "gap(b_det)", "β"]
show(rep, {"b*": "{:.0f}", "Π*": "{:.2f}", "b̄": "{:.0f}", "Π(b̄)": "{:.2f}", "gap(b̄)": "{:.2%}",
           "gap(b̄) paper": "{:.1%}", "b_det": "{:.0f}", "gap(b_det)": "{:.2%}", "β": "{:.2f}"})
fig1 = plotting.fig1_reproduction(rows)
plotting.save(fig1, "fig1_reproduction");
''')

md("""
### 3 · When does the quantile rule win? *(2 min)*

**Theorem 1** (paper §5): the newsvendor rule is near-optimal when $\\mathrm{CV} \\gg 1/\\sqrt{E}$ (uncertainty-dominated); square-root safety staffing matters when $\\mathrm{CV} \\ll 1/\\sqrt{E}$ ($E$ = offered load in Erlangs).
""")
code(r'''
if RECOMPUTE:
    E_vals, cv_vals = np.geomspace(10, 3000, 15), np.geomspace(0.01, 0.6, 15)
    g = regime_grid(E_vals, cv_vals, "gamma")
    grid, kappa = g, boundary_analysis(g)["kappa_median"]
else:
    g = load_regime_grid()
    grid = {"E": g["E"], "cv": g["cv"], "log_ratio": g["gamma_log_ratio"]}
    kappa = g["meta"]["boundary"]["gamma"]["kappa_median"]

# where AI sits: concurrency today (day-ahead CV) and daily energy in 2030 (CV of the PJM model's P_T)
cv2030 = pl.cv_2030_range(sim)
ai = cox.ai_regime_numbers(cv_2030=cv2030)
boxes = cox.regime_boxes(ai)
print(f"Cheaper rule switches at CV ≈ {kappa:.2f}/√E (Theorem 1 says the boundary scales like 1/√E).")
fig2 = plotting.fig2_regime_map(grid, kappa, boxes)
plotting.save(fig2, "fig2_regime_map");
''')

md("""
### 4 · AI queries as a Cox process *(2 min)*

Daily queries $N \\mid \\xi \\sim \\text{Poisson}(\\bar n\\,\\xi)$ with an uncertain rate multiplier $\\xi$; daily energy $S = \\sum_{i\\le N} X_i$:
$$\\mathrm{CV}^2(S) = \\frac{e^{\\sigma^2}}{\\bar n} + \\mathrm{CV}^2_{\\text{rate}}$$
Per-query noise dies like $1/\\sqrt{\\bar n}$; rate uncertainty is a floor. Crossover: $\\bar n^* = e^{\\sigma^2}/\\mathrm{CV}^2_{\\text{rate}}$.
""")
code(r'''
CV_RATE = 0.5   # ASSUMPTION for the illustration (a parameter)
if RECOMPUTE:
    cs = cox.simulate_cv_curve(np.logspace(1, 6, 11), [1.0, 2.0, 3.0], CV_RATE)
else:
    with np.load(aicap.OUTPUTS / "cache" / "cox_sim.npz") as z:
        cs = {k: z[k] for k in z.files}
c, dl = ai["concurrency"], ai["daily"]
lo, hi = ai["ratio_day_ahead_vs_concurrency"], ai["ratio_2030_vs_daily"]
regime = pd.DataFrame({
    "offered load E": [f"{c['R']:.1e} ({c['R_lo']:.0e}–{c['R_hi']:.0e})", f"{dl['E_eff_lo']:.0e}–{dl['E_eff_hi']:.0e}"],
    "per-query noise 1/√E": [f"{c['noise']:.1e}", f"{dl['noise_lo']:.0e}–{dl['noise_hi']:.0e}"],
    "rate CV": ["0.05–0.2 (day-ahead, ASSUMPTION)", f"{cv2030[0]:.1f}–{cv2030[1]:.1f} (PJM model, P_T)"],
    "CV·√E  (switch at ≈1.4)": [f"{lo[0]:.0f}–{lo[1]:.0f}", f"{hi[0]:.0e}–{hi[1]:.0e}"],
}, index=["today: requests in service R (2.5e9/day × 2–60 s)", "2030: daily energy E_eff = n̄/e^(σ²) (10¹⁰–2×10¹⁰/day, σ 1–3)"])
show(regime)
print("crossover n̄* (CV_rate = 0.5):", {f"σ={s:g}": round(cox.crossover_n(s, CV_RATE)) for s in (1, 2, 3)})
fig3 = plotting.fig3_cox_floor(cs, CV_RATE, cox.cv_daily_energy, cox.crossover_n)
plotting.save(fig3, "fig3_cox_floor");
''')

md("""
### 5 · New PJM summer-peak load ΔQ and the capacity to buy *(3 min)*

$$P_T = \\frac{N_0\\,G_T\\,m_c\\,\\mathrm{mix}_T}{H_T}\\cdot\\frac{(1+\\tau)\\,s_{US}\\,s_{PJM}\\,\\kappa}{24\\ \\text{h}},\\qquad \\Delta Q = P_T - P_0$$
Newsvendor reading *(an analogy, not PJM's method)*: buy $Q$ with $P(\\Delta Q > Q) = c_{\\text{cap}}/c_{\\text{short}}$.

**Fermi ladder**: the median path from ChatGPT today to PJM in 2030/31, one factor per step.
""")
code(r'''
ladder = pd.DataFrame(pl.fermi_ladder(T=5, structure="matures")).set_index("step")
show(ladder, {"factor": "×{:.3g}", "MW": "{:,.0f}"})
''')
code(r'''
EFFICIENCY = "matures"   # ← flip to "compounding" live and re-run this cell

tbl = {}
for year, dy in DELIVERY_YEARS.items():
    d = sim[(year, EFFICIENCY)]["dQ"]
    q = pl.quantile_table(d, probs=(0.10, 0.50, 0.90, 0.95), n_boot=50)
    tbl[f"DY {dy['label']}"] = {"P10": q["P10"], "median": q["P50"], "P90": q["P90"], "P95": q["P95"],
                                "mean": q["mean"], "P(ΔQ > 30 GW)": q["p_exceed"],
                                "cost of uncertainty P90 − median": q["P90"] - q["P50"]}
print(f"ΔQ in GW, efficiency = {EFFICIENCY!r}, {N_DRAWS:,} draws (MC standard errors in outputs/summary.json)")
show(pd.DataFrame(tbl).T, {"P(ΔQ > 30 GW)": "{:.3f}", **{k: "{:.2f}" for k in
     ["P10", "median", "P90", "P95", "mean", "cost of uncertainty P90 − median"]}})

d30 = sim[(2030, EFFICIENCY)]["dQ"]
cap = pd.DataFrame({"cost ratio c_cap/c_short": [f"{r:.2f}" for r in COST_RATIOS],
                    "buy Q = quantile": [f"P{round(100 * (1 - r))}" for r in COST_RATIOS],
                    "Q (GW)": [pl.quantile_capacity(d30)[f"{r:.2f}"] for r in COST_RATIOS]}).set_index("cost ratio c_cap/c_short")
show(cap, {"Q (GW)": "{:.1f}"})
pos = pl.position_of(d30, PJM_DC_GROWTH_GW)
print(f"PJM's ~30 GW (all data centres, not AI alone) sits at the {pos['quantile']:.1%} quantile "
      f"→ implied c_cap/c_short ≈ {pos['implied_cost_ratio']:.3f}")
''')
code(r'''
fig4 = plotting.fig4_handoff(sim[(2030, "matures")]["dQ"], sim[(2030, "compounding")]["dQ"], COST_RATIOS,
                             pjm_level=PJM_DC_GROWTH_GW, pjm_peak=PJM_SUMMER_PEAK_2026_GW)
plotting.save(fig4, "fig4_handoff");
''')

md("""
**Policy hooks.** *Better forecasting*: halve the uncertainty of growth and query mix and see how much capacity is no longer needed. *Flexible load*: curtailable data centres lower the cost of a shortfall, so the fractile and the capacity both drop.
""")
code(r'''
from aicap.params import scaled_priors
half = pl.simulate(n=N_DRAWS, seed=SEED, priors=scaled_priors(["growth", "mix"], 0.5))
base, better = pl.cost_of_uncertainty(d30), pl.cost_of_uncertainty(half[(2030, EFFICIENCY)]["dQ"])
print(f"Cost of uncertainty (P90 − median), DY 2030/31: {base['gap']:.1f} GW → {better['gap']:.1f} GW "
      f"with half the growth and mix uncertainty ({1 - better['gap'] / base['gap']:.0%} less)")
q90, q80 = np.quantile(d30, [0.90, 0.80])
print(f"Flexible load: halving the shortage cost moves the fractile from P90 to P80: {q90:.1f} GW → {q80:.1f} GW")
''')

md("""
### 6 · Handoff to presenter 3 *(1 min)*

Samples of ΔQ (GW) for DY 2028/29 and 2030/31, both efficiency structures. For a convex price curve $p$, Jensen gives $E[p(\\Delta Q)] \\ge p(E[\\Delta Q])$, so **push every sample through the price function**; never just the median.
""")
code(r'''
summary_d = pl.summarize(sim, n_boot=50, verbose=False)
out = pl.write_handoff(sim, summary_d, run_info())
for f in sorted(out.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")
print("\nBridge: the GW I hand over is a distribution; the capacity bought is a high quantile of it, "
      "and presenter 3 turns each draw into a price.")
''')

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("notebook written:", sys.argv[1], len(cells), "cells")
