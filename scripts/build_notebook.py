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

**Presenter 2: code demo (≈13 min).** Presenter 1's Wh per query → a distribution of AI electricity use and of new PJM summer-peak load ΔQ (GW) → why the capacity to buy is a *quantile* of that distribution, not its median → energy samples handed to presenter 3, who turns them into water.

| Part | What | Time |
|---|---|---|
| **I · Derivation** | the newsvendor first-order condition: capacity is a quantile | 3 min |
| **II · Demo** (cells 1–5) | presenter 1's Wh → the paper → when the quantile rule wins → where AI sits → PJM ΔQ and the capacity to buy | 7 min |
| **III · Pseudocode** | the whole method in eight lines, then run once more | 1.5 min |
| **6 · Bridge** | energy to presenter 3 (water) | 1 min |

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

sim = pl.simulate(n=N_DRAWS, seed=SEED)   # 200,000 parameter draws (0.05 s); used in Parts I–III and cell 6
print(run_info())
''')

md(r"""
## Part I · The heart: the newsvendor first-order condition *(3 min)*

**Problem.** Choose capacity $Q$ **before** demand $X$ is known ($X$ random with CDF $F$ and density $f$). Each unit of capacity costs $c$ whether it is used or not; each unit of demand above capacity costs $u > c$.
$$\bar\Pi(Q) = c\,Q + u\,E\big[(X-Q)^+\big] = c\,Q + u\int_Q^\infty (x-Q)\,f(x)\,dx$$

**1 · Differentiate** (Leibniz rule; the boundary term $(Q-Q)\,f(Q)$ vanishes):
$$\frac{d}{dQ}\,E\big[(X-Q)^+\big] = -\int_Q^\infty f(x)\,dx = -P(X>Q) \qquad\Longrightarrow\qquad \bar\Pi'(Q) = c - u\,P(X>Q)$$

**2 · Convexity:** $\bar\Pi''(Q) = u\,f(Q) \ge 0$, so the stationary point is the global minimum.

**3 · First-order condition = the critical quantile:**
$$\boxed{\;P(X > Q^*) = \frac{c}{u} =: \alpha \quad\Longleftrightarrow\quad Q^* = F^{-1}(1-\alpha)\;}$$
(if $c \ge u$, buy nothing). *Marginal reading:* one more unit costs $c$ for sure and saves $u$ only if it gets used, which happens with probability $P(X > Q)$; keep adding until the two balance. *Classic form:* overage cost $c_o = c$ and underage cost $c_u = u - c$ give $F(Q^*) = c_u/(c_u + c_o)$, so the median is optimal only when $c_o = c_u$, i.e. $u = 2c$.

**In the paper** (eqs. 2–3): $X$ is the arrival rate $\Lambda$ and capacity is $\mu b$. Fluid approximation: the unserved rate $(\Lambda - \mu b)^+$ abandons (penalty $p$ each) after waiting in a queue of about $(\Lambda - \mu b)^+/\gamma$ (holding cost $h$), so $u = p + h/\gamma$ and
$$P(\Lambda > \mu\bar b) = \frac{c}{\mu\,(p + h/\gamma)} = \frac{1/3}{1 \cdot (1 + 1/3)} = \frac14 \quad\Longrightarrow\quad \bar b = \text{75th percentile of } \Lambda/\mu .$$
The fluid step throws away the Poisson noise, of order $\sqrt{E}$. Part II asks when that is safe.

**For AI load:** $X = \Delta Q$ (new PJM peak load, GW), $c = c_\text{cap}$, $u = c_\text{short}$: buy the $(1 - c_\text{cap}/c_\text{short})$ quantile of ΔQ.
""")
code(r'''
ALPHA = 0.10   # c_cap / c_short  ← try 0.25 or 0.50 live
from aicap.erlang_a import SECTION4
print(f"Paper §4: α = c/(μ(p + h/γ)) = {SECTION4.fractile:.2f}  →  b̄ = {100 * (1 - SECTION4.fractile):.0f}th percentile of Λ/μ")
d30 = sim[(2030, "matures")]["dQ"]
chk = pl.newsvendor_check(d30, ratios=(ALPHA,))[f"{ALPHA:.2f}"]
print(f"AI, DY 2030/31, α = {ALPHA:.2f}")
lines = [(f"brute force: minimizer of α·Q + E(ΔQ − Q)⁺ over all {N_DRAWS:,} draws", chk["argmin_Q"]),
         (f"first-order condition: the {1 - ALPHA:.0%} quantile of ΔQ", chk["quantile_Q"])]
for text, v in lines:
    print(f"  {text + ':':<{max(len(t) for t, _ in lines) + 1}} {v:6.2f} GW")
print(f"  buying the median ({chk['median']:.1f} GW) instead raises the expected cost by {chk['excess_median']:.0%}")
fig0 = plotting.fig0_newsvendor_foc(d30, ALPHA)   # run_all.py saves this figure (α = 0.10) for slides
''')

md("""
## Part II · Demo *(7 min)*: does the fluid step hold for AI, and what does the quantile come to?

### 1 · From presenter 1: energy per query *(½ min)*

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
### 2 · The paper: is dropping the queue noise safe? *(1½ min)*

Erlang-A queue (M/M/b+M): Poisson arrivals at rate $\\Lambda$ (itself random), $b$ servers, service rate $\\mu=1$, abandonment $\\gamma=3$. The **exact** cost keeps the Poisson noise; its minimizer $b^*$ is found by search:
$$\\Pi(b) = c\\,b + (h + p\\gamma)\\,E\\big[(N-b)^+\\big]$$
Against it: the newsvendor rule $\\bar b$ from Part I (75th percentile of $\\Lambda$, Poisson noise ignored) and the known-rate rule, the best $b$ if $\\Lambda$ were exactly its mean ($\\approx E + \\beta\\sqrt{E}$).
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
### 3 · When does the quantile rule win? *(1½ min)*

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
### 4 · AI queries as a Cox process *(1 min)*

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
### 5 · New PJM summer-peak load ΔQ and the capacity to buy *(2½ min)*

$$P_T = \\frac{N_0\\,G_T\\,m_c\\,\\mathrm{mix}_T}{H_T}\\cdot\\frac{(1+\\tau)\\,s_{US}\\,s_{PJM}\\,\\kappa}{24\\ \\text{h}},\\qquad \\Delta Q = P_T - P_0$$
Part I with $X = \\Delta Q$ *(an analogy, not PJM's method)*: buy $Q$ with $P(\\Delta Q > Q) = c_{\\text{cap}}/c_{\\text{short}}$.

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
**Policy hooks** *(optional, if time)*. *Better forecasting*: halve the uncertainty of growth and query mix and see how much capacity is no longer needed. *Flexible load*: curtailable data centres lower the cost of a shortfall, so the fractile and the capacity both drop.
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

md(r"""
## Part III · The method as pseudocode *(1½ min)*

```text
SIZE-CAPACITY(priors, horizon T, cost ratio α = c_cap / c_short, draws n)        code (src/aicap/)

  ▸ Step 1: is the newsvendor (fluid) step safe?                         Part II, cells 2–4
  1   E ← offered load;  CV ← coefficient of variation of the rate
  2   if CV·√E ≲ 1.4:  return the square-root rule  E + β√E               capacity_rules.known_rate_capacity

  ▸ Step 2: sample what we do not know                                   Part II, cell 5
  3   for i = 1, …, n:                                                    pjm_load.simulate
  4       θᵢ ← one independent draw from each prior                       pjm_load.sample_params
  5       ΔQᵢ ← P_T(θᵢ) − P_0(θᵢ)                                          pjm_load.delta_q

  ▸ Step 3: the first-order condition                                    Part I
  6   Q* ← empirical (1 − α) quantile of ΔQ₁, …, ΔQₙ      ▹ P(ΔQ > Q*) = α  pjm_load.quantile_capacity
  7   SE ← std. dev. of line 6 over B bootstrap resamples of ΔQ           pjm_load.quantile_table
  8   return Q*, SE
```
For AI, line 2 never fires: CV·√E sits well above the switch at ≈ 1.4 (cells 3–4). Below, lines 3–8 run once more with fresh random numbers.
""")
code(r'''
def size_capacity(alphas, T=5, n=N_DRAWS, seed=SEED + 1, B=50):
    """Lines 3-8 of SIZE-CAPACITY for several cost ratios at once, with fresh random numbers."""
    rng = np.random.default_rng(seed)
    theta = pl.sample_params(n, rng)                                        # lines 3-4
    dQ = pl.delta_q(theta, T)["dQ"]                                         # line 5
    probs = 1 - np.asarray(alphas)
    Q = np.quantile(dQ, probs)                                              # line 6: the first-order condition
    boot = np.array([np.quantile(dQ[rng.integers(0, n, n)], probs) for _ in range(B)])  # line 7
    return Q, boot.std(axis=0, ddof=1)                                      # line 8

ratio_lo = ai["ratio_day_ahead_vs_concurrency"][0]
print(f"Line 2: CV·√E for AI is at least {ratio_lo:.0f}, against a switch at ≈ {kappa:.1f}, so continue to line 3.")
Q, se = size_capacity(COST_RATIOS)
ref = pl.quantile_capacity(sim[(2030, "matures")]["dQ"])
res = pd.DataFrame({"buy": [f"P{round(100 * (1 - a))}" for a in COST_RATIOS], "Q* (GW), fresh draws": Q, "± SE": se,
                    "Q* (GW), Part II draws": [ref[f"{a:.2f}"] for a in COST_RATIOS]},
                   index=pd.Index([f"{a:.2f}" for a in COST_RATIOS], name="α = c_cap/c_short"))
show(res, {"Q* (GW), fresh draws": "{:.1f}", "± SE": "{:.1f}", "Q* (GW), Part II draws": "{:.1f}"})
''')

md("""
### 6 · Bridge to presenter 3: from energy to water *(1 min)*

The water footprint is driven by **energy**, not peak load. Li et al. (2025): operational water $= \\sum_t e_t\\,[\\rho_{s1,t} + \\theta_t\\,\\rho_{s2,t}]$, with $e_t$ the server energy, $\\rho_{s1}$ the on-site WUE (cooling), $\\theta$ the PUE and $\\rho_{s2}$ the off-site EWIF (power plants).
Water is linear in energy, so *expected* water needs only expected energy, but whether a **water budget** holds depends on the whole distribution: the same quantile logic as capacity.
""")
code(r'''
E = pl.energy_tables(sim, n_boot=0)   # annualized TWh/yr; facility-level energy (see the handoff README)
g = pl.TWH_TO_GWH_PER_DAY
rows = {}
for r in pl.REGIONS:
    b, e = E["baseline_2025"][r], E["cases"]["2030_matures"][r]["E"]
    rows[r] = {"2025 median (TWh/yr)": b["P50"], "2030 median (TWh/yr)": e["P50"], "2030 P90 (TWh/yr)": e["P90"],
               "2030 median (GWh/day)": e["P50"] * g, "2030 P90 (GWh/day)": e["P90"] * g}
show(pd.DataFrame(rows).T, {c: "{:,.1f}" for c in pd.DataFrame(rows).T.columns})
print("Bridge: I sized capacity before demand is known: a quantile of the load distribution. Presenter 3 takes the same\n"
      "energy draws and asks what they cost in water, on site for cooling and off site at the power plant, and how\n"
      "choosing where the computing runs can keep that within a budget.")
''')
code(r'''
summary_d = pl.summarize(sim, n_boot=20, verbose=False, energy_n_boot=0)
out = pl.write_handoff(sim, summary_d, run_info(), csv=False)   # handoff_summary.json + README.md
''')
code(r'''
pl.write_energy_csvs(sim, run_info())   # energy samples (TWh/yr) for presenter 3
pl.write_peak_csvs(sim, run_info())     # peak-load samples (GW)
for f in sorted(out.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")
''')

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("notebook written:", sys.argv[1], len(cells), "cells")
