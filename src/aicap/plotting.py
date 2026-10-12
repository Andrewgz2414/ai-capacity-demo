"""Shared figure style and the five demo figures (Fig. 0: the first-order condition; Figs. 1-4).

Projector style: fonts >= 14 pt (12.5 pt only for secondary annotations), data lines >= 2.5 pt,
solid hairline grid, light surface. Colours come from a validated colour-blind-safe reference
palette (blue / orange categorical slots, a one-hue blue ramp for ordered series, blue <-> red
diverging with a gray midpoint); one accent colour marks the point of each plot and comparison
series are gray. Every figure is saved as PNG (200 dpi) and SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Palette (light surface)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # primary text
INK2 = "#52514e"  # secondary text
MUTED = "#898781"  # de-emphasised series
GRID = "#e1e0d9"  # hairline grid
AXIS = "#c3c2b7"  # baseline / spines / reference lines
ACCENT = "#2a78d6"  # categorical slot 1 (blue)
SECOND = "#eb6834"  # categorical slot 2 (orange)
BLUE_RAMP = {100: "#cde2fb", 250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281"}
NEUTRAL_MID = "#f0efec"
RED = "#e34948"
DARK_RED = "#a3282a"
LIGHT_RED = "#f8d3d0"

# Diverging map: blue (negative) <-> gray midpoint <-> red (positive), equal arms.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", [BLUE_RAMP[650], BLUE_RAMP[450], BLUE_RAMP[250], NEUTRAL_MID, "#f2a7a3", RED, DARK_RED], N=256)

def _fig_dir() -> Path:
    from . import OUTPUTS
    return OUTPUTS / "figures"
LABEL_BOX = dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        # DejaVu Sans ships with matplotlib and covers every glyph used here (√, ², ₀, macrons, arrows)
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "legend.fontsize": 14, "figure.titlesize": 18,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 1.2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 5, "ytick.major.size": 5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2.5, "lines.markersize": 9,
        "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
        "svg.fonttype": "path",
    })


def save(fig, name: str, outdir: Path | None = None) -> list[Path]:
    outdir = Path(outdir) if outdir else _fig_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{name}.png", outdir / f"{name}.svg"]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    return paths


def _pct(x: float) -> str:
    if abs(x) < 5e-5:
        return "0.00%"
    return f"{100 * x:.2f}%" if abs(x) < 0.1 else f"{100 * x:.1f}%"


def _boxed_legend(ax, *args, **kw):
    kw.setdefault("frameon", True)
    leg = ax.legend(*args, facecolor=SURFACE, edgecolor=AXIS, framealpha=0.95, **kw)
    return leg


# ---------------------------------------------------------------------------
# Figure 0: the newsvendor first-order condition on the Delta Q samples
# ---------------------------------------------------------------------------


def fig0_newsvendor_foc(dq, ratio: float = 0.10, label: str = "2030/31", x_max: float | None = None):
    """Left: marginal cost c_cap against marginal saving c_short P(dQ > Q); they cross at the quantile.
    Right: the expected cost c_cap Q + c_short E(dQ - Q)^+ (units of c_short), minimized at the same Q."""
    from .pjm_load import newsvendor_cost_curve

    apply_style()
    dq = np.sort(np.asarray(dq, dtype=float))
    q_star, med = float(np.quantile(dq, 1 - ratio)), float(np.median(dq))
    x_max = x_max or 10 * np.ceil(max(60.0, 1.5 * q_star) / 10)
    x = np.linspace(0.0, x_max, 601)
    ccdf = 1 - np.searchsorted(dq, x, side="right") / dq.size
    pq = round(100 * (1 - ratio))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))

    ax = axes[0]
    ax.fill_between(x, ratio, ccdf, where=ccdf >= ratio, color=BLUE_RAMP[100], lw=0, zorder=1)
    ax.plot(x, ccdf, color=ACCENT, lw=3.2, zorder=3, label=r"saving: $c_\mathrm{short}\,P(\Delta Q > Q)$")
    ax.axhline(ratio, color=SECOND, lw=3.0, zorder=2, label=r"cost: $c_\mathrm{cap}$")
    ax.plot([q_star], [ratio], marker="o", ms=12, color=INK, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(f"Q* = P{pq} = {q_star:.1f} GW", (q_star, ratio), xytext=(12, 12), textcoords="offset points",
                ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK, bbox=LABEL_BOX)
    ax.text(0.97, 0.52 if ratio < 0.35 else 0.28,  # above the cost line, or below it for high ratios
            "left of Q*:  saving > cost, add capacity\nright of Q*:  cost > saving, stop",
            transform=ax.transAxes, ha="right", va="center", fontsize=13.5, color=INK2, bbox=LABEL_BOX, zorder=4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(r"per GW of capacity  (units of $c_\mathrm{short}$)")
    ax.set_title("Marginal analysis: one more GW", loc="left", pad=10)
    _boxed_legend(ax, loc="upper right", fontsize=13.5)

    ax = axes[1]
    cost = newsvendor_cost_curve(dq, ratio)
    y = cost(x)
    c_star, c_med = float(cost(q_star)), float(cost(med))
    ax.plot(x, y, color=ACCENT, lw=3.2, zorder=3)
    ax.plot([q_star], [c_star], marker="o", ms=12, color=INK, mec=SURFACE, mew=2, zorder=5)
    left = q_star < 0.3 * x_max  # keep the label inside the axes
    ax.annotate(f"minimum at P{pq} = {q_star:.1f} GW", (q_star, c_star), xytext=(-12 if left else 0, -16),
                textcoords="offset points", ha="left" if left else "center", va="top", fontsize=14,
                fontweight="bold", color=INK)
    ax.plot([med], [c_med], marker="o", ms=11, color=MUTED, mec=SURFACE, mew=2, zorder=4)
    ax.annotate(f"median {med:.1f} GW:\nexpected cost +{100 * (c_med / c_star - 1):.0f}%", (med, c_med),
                xytext=(14, 8), textcoords="offset points", ha="left", va="bottom", fontsize=13.5, color=INK2,
                bbox=LABEL_BOX)
    lo, hi = float(y.min()), float(y.max())
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_ylabel(r"expected cost  (units of $c_\mathrm{short}$)")
    ax.set_title(r"Expected cost  $c_\mathrm{cap}Q + c_\mathrm{short}\,E(\Delta Q - Q)^+$", loc="left", pad=10)

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_xlabel(f"capacity Q  (GW of new PJM peak load, DY {label})")
        ax.axvline(q_star, color=AXIS, lw=1.3, zorder=0)
    fig.suptitle(rf"First-order condition: $c_\mathrm{{cap}} = c_\mathrm{{short}}\,P(\Delta Q > Q^*)$, "
                 rf"so with $c_\mathrm{{cap}}/c_\mathrm{{short}}$ = {ratio:.2f} buy the P{pq}",
                 x=0.5, y=0.975, fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.84, wspace=0.28)
    return fig


# ---------------------------------------------------------------------------
# Figure 1: reproduction of the paper's section 4
# ---------------------------------------------------------------------------


def fig1_reproduction(rows: list[dict]):
    """rows: reproduction rows for U[25,50], U[50,100], U[200,400] and lambda = 37.5, 75, 300."""
    apply_style()
    random_rows = sorted([r for r in rows if r["cv"] > 0], key=lambda r: r["E"])
    fixed_rows = sorted([r for r in rows if r["cv"] == 0], key=lambda r: r["E"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    panels = [(axes[0], random_rows, "Rate uncertain: Λ ~ U[a, 2a]  (CV 19%)"),
              (axes[1], fixed_rows, "Rate known: λ fixed")]
    for ax, rr, title in panels:
        E = np.array([r["E"] for r in rr])
        nv = 100 * np.array([r["gap_bar"] for r in rr])
        kr = 100 * np.array([r["gap_det"] for r in rr])
        paper = 100 * np.array([r["paper_gap_bar"] for r in rr])
        ax.plot(E, kr, color=MUTED, marker="s", lw=2.5, ms=9, zorder=3,
                label="known-rate rule  b = argmin C(b; E[Λ])")
        ax.plot(E, nv, color=ACCENT, marker="o", lw=3.2, ms=10, zorder=4,
                label="newsvendor rule  b̄ = 75th pct of Λ")
        ax.plot(E, paper, ls="none", marker="o", ms=15, mfc="none", mec=ACCENT, mew=1.8, zorder=5,
                label="newsvendor, paper's Tables 2–3")
        for x, y in zip(E, nv):  # labels under the points keep clear of the paper markers above
            ax.annotate(_pct(y / 100), (x, y), xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top", fontsize=14, color=INK, fontweight="bold")
        if ax is axes[0]:
            ax.annotate(_pct(kr[-1] / 100), (E[-1], kr[-1]), xytext=(-12, 8), textcoords="offset points",
                        ha="right", va="bottom", fontsize=14, color=INK2)
        else:
            ax.annotate("known-rate rule is optimal (gap 0)", (E[1], 0), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom", fontsize=13, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(E)
        ax.set_xticklabels([f"{e:g}" for e in E])
        ax.minorticks_off()
        ax.set_xlim(E[0] / 1.35, E[-1] * 1.35)
        ax.set_title(title, loc="left", color=INK, pad=10)
        ax.set_xlabel("mean offered load  E[Λ]/μ  (Erlangs)")
    axes[0].set_ylabel("optimality gap  Π(rule)/Π* − 1  (%)")
    axes[0].set_ylim(-1.6, 12.5)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 0]  # newsvendor, paper's newsvendor, known-rate
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="center", ncol=3,
               bbox_to_anchor=(0.5, 0.875), fontsize=13, handlelength=2.0, columnspacing=1.4)
    nv_unc = max(r["gap_bar"] for r in random_rows)
    nv_det = [r["gap_bar"] for r in fixed_rows]
    fig.suptitle(f"Rate uncertain: newsvendor rule within {_pct(nv_unc)} of optimal.  "
                 f"Rate known: {_pct(min(nv_det))}–{_pct(max(nv_det))} off",
                 x=0.5, y=0.975, fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.76, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: regime map
# ---------------------------------------------------------------------------


def _edges(v):
    v = np.asarray(v, float)
    mids = np.sqrt(v[:-1] * v[1:])
    return np.concatenate(([v[0] ** 2 / mids[0]], mids, [v[-1] ** 2 / mids[-1]]))


def fig2_regime_map(grid: dict, kappa: float, ai: dict, paper_cases=((37.5, 0.192), (75, 0.192), (300, 0.192))):
    """grid: E, cv, log_ratio (E x cv); kappa: fitted switch constant; ai: boxes for the context panel,
    {"concurrency": (E_lo, E_hi, cv_lo, cv_hi), "daily_2030": (E_lo, E_hi, cv_lo, cv_hi),
     "labels": {"concurrency": str, "daily_2030": str}}."""
    apply_style()
    E, cv, lr = grid["E"], grid["cv"], np.clip(grid["log_ratio"], -3, 3)
    fig = plt.figure(figsize=(17.5, 8.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.18, 0.045, 0.36, 1.0], wspace=0.04,
                          left=0.055, right=0.985, bottom=0.11, top=0.80)
    ax, cax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
    mesh = ax.pcolormesh(_edges(E), _edges(cv), lr.T, cmap=DIVERGING, vmin=-3, vmax=3, shading="flat",
                         edgecolors=SURFACE, linewidth=0.6)
    ee = np.geomspace(_edges(E)[0], _edges(E)[-1], 200)
    ax.plot(ee, 1 / np.sqrt(ee), color=INK, lw=3, label="CV = 1/√E  (Theorem 1)")
    ax.plot(ee, kappa / np.sqrt(ee), color=INK, lw=2.2, ls=(0, (6, 3)), label=f"fitted switch  CV = {kappa:.1f}/√E")
    pe, pc = zip(*paper_cases)
    ax.plot(pe, pc, ls="none", marker="o", ms=13, mfc="white", mec=INK, mew=2, label="paper's §4 cases U[a, 2a]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(_edges(E)[0], _edges(E)[-1])
    ax.set_ylim(_edges(cv)[0], _edges(cv)[-1])
    ax.grid(False)
    ax.set_xlabel("offered load  E = E[Λ]/μ  (Erlangs)")
    ax.set_ylabel("rate uncertainty  CV = sd(Λ)/E[Λ]  (ratio)")
    ax.text(0.97, 0.96, "newsvendor (quantile)\nrule cheaper", transform=ax.transAxes, ha="right", va="top",
            color="white", fontsize=15, fontweight="bold")
    ax.text(0.03, 0.04, "√E safety-staffing\nrule cheaper", transform=ax.transAxes, ha="left", va="bottom",
            color="white", fontsize=15, fontweight="bold")
    cb = fig.colorbar(mesh, cax=cax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.ax.set_yticklabels(["1000×", "100×", "10×", "equal", "10×", "100×", "1000×"])
    cb.set_label("gap(b̄) / gap(b_det)  (log scale)\nblue: newsvendor gap smaller · red: known-rate gap smaller",
                 fontsize=12.5)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title("Gamma-distributed rate, 15 × 15 grid, exact Erlang-A", loc="left", fontsize=15)
    ax.annotate("AI at grid scale", xy=(1.0, 0.62), xycoords="axes fraction", xytext=(0.70, 0.62),
                textcoords="axes fraction", ha="center", va="center", fontsize=14, color=INK, fontweight="bold",
                bbox=LABEL_BOX, arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=20))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.875), fontsize=13.5,
               handlelength=2.4, columnspacing=2.0)

    # zoomed-out context panel
    x = np.geomspace(3, 3e11, 300)
    y_lo, y_hi = 2e-6, 12
    ax2.fill_between(x, kappa / np.sqrt(x), y_hi, color=BLUE_RAMP[100], lw=0)
    ax2.fill_between(x, y_lo, kappa / np.sqrt(x), color=LIGHT_RED, lw=0)
    ax2.plot(x, 1 / np.sqrt(x), color=INK, lw=3)
    ax2.plot(x, kappa / np.sqrt(x), color=INK, lw=2.2, ls=(0, (6, 3)))
    e0, e1, c0, c1 = _edges(E)[0], _edges(E)[-1], _edges(cv)[0], _edges(cv)[-1]
    ax2.add_patch(plt.Rectangle((e0, c0), e1 - e0, c1 - c0, fill=False, ec=INK2, lw=2))
    ax2.text(e0, c1 * 1.5, "map at left", fontsize=13, color=INK2, va="bottom", ha="left")
    for key in ("concurrency", "daily_2030"):
        x0, x1, y0, y1 = ai[key]
        ax2.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=ACCENT, ec=BLUE_RAMP[650], lw=2))
    labels = ai.get("labels", {})
    cx0, cx1, cy0, cy1 = ai["concurrency"]
    ax2.text(cx1 * 2.0, np.sqrt(cy0 * cy1), labels.get("concurrency", "AI today, day-ahead"),
             ha="left", va="center", fontsize=12.5, color=INK, bbox=LABEL_BOX)
    dx0, dx1, dy0, dy1 = ai["daily_2030"]
    ax2.text(np.sqrt(dx0 * dx1), dy0 / 1.7, labels.get("daily_2030", "AI in 2030, daily energy"),
             ha="center", va="top", fontsize=12.5, color=INK, bbox=LABEL_BOX)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(3, 3e11)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel("offered load  E  (Erlangs; queries/e^(σ²) for daily energy)")
    ax2.set_ylabel("CV of the rate  (ratio)")
    ax2.set_title("Zoomed out: where AI sits", loc="left", fontsize=15)
    ax2.text(0.97, 0.47, "uncertainty-dominated:\nquantile (newsvendor)", transform=ax2.transAxes, ha="right",
             va="center", fontsize=14, color=BLUE_RAMP[650], fontweight="bold")
    ax2.text(0.03, 0.04, "variability-dominated:\n√E safety staffing", transform=ax2.transAxes, ha="left",
             va="bottom", fontsize=14, color=DARK_RED, fontweight="bold")
    fig.suptitle(f"The quantile rule wins above CV ≈ {kappa:.1f}/√E, and AI at grid scale sits far above that line",
                 fontsize=18, fontweight="bold", y=0.975)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Cox process, CV of daily energy vs volume
# ---------------------------------------------------------------------------


def fig3_cox_floor(sim: dict, cv_rate: float, theory_fn, crossover_fn, n_sim_max: float = 1e6,
                   ai_band=(1e10, 2e10), bench_band=(100, 1000)):
    """sim: output of cox.simulate_cv_curve; theory_fn(nbar, sigma, cv_rate); crossover_fn(sigma, cv_rate)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    colors = [BLUE_RAMP[250], BLUE_RAMP[450], BLUE_RAMP[650]]
    x_sim = np.geomspace(5, n_sim_max, 200)
    x_ext = np.geomspace(n_sim_max, 3e11, 200)
    ax.axvspan(*bench_band, color=GRID, lw=0, zorder=0)
    ax.axvspan(*ai_band, color=BLUE_RAMP[100], lw=0, zorder=0)
    ax.text(np.sqrt(bench_band[0] * bench_band[1]), 55, "typical\nbenchmark\n(100–1,000)", ha="center", va="top",
            fontsize=12.5, color=INK2)
    ax.text(np.sqrt(ai_band[0] * ai_band[1]), 0.315, "AI at grid scale\n≈10¹⁰ queries/day", ha="center",
            va="bottom", fontsize=13, color=BLUE_RAMP[650], fontweight="bold", bbox=LABEL_BOX)
    ax.axhline(cv_rate, color=SECOND, lw=3.4, zorder=2)
    ax.text(1.2e9, cv_rate * 0.9, f"floor = CV of the rate = {cv_rate:g}", color=SECOND, fontsize=14,
            fontweight="bold", va="top", ha="right")
    for i, s in enumerate(sim["sigmas"]):
        c = colors[i]
        ax.plot(x_sim, theory_fn(x_sim, s, cv_rate), color=c, lw=3, zorder=3, label=f"σ = {s:g}  (theory)")
        ax.plot(x_ext, theory_fn(x_ext, s, cv_rate), color=c, lw=3, ls=(0, (5, 3)), zorder=3)
        filled = s < 2.5
        ax.plot(sim["nbar"], sim["cv_hat"][i], ls="none", marker="o", ms=10, zorder=4,
                mfc=c if filled else SURFACE, mec=BLUE_RAMP[650] if filled else c, mew=2)
        ns = crossover_fn(s, cv_rate)
        ax.plot([ns], [theory_fn(ns, s, cv_rate)], marker="D", ms=10, color=SECOND, mec=SURFACE, mew=1.5, zorder=5)
        ax.annotate(f"n̄* ≈ {ns:,.0f}", (ns, theory_fn(ns, s, cv_rate)), xytext=(9, 7), textcoords="offset points",
                    fontsize=13, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], ls="none", marker="o", ms=10, mfc=MUTED, mec=INK2),
                Line2D([], [], ls="none", marker="o", ms=10, mfc=SURFACE, mec=MUTED, mew=2),
                Line2D([], [], color=MUTED, lw=3, ls=(0, (5, 3))),
                Line2D([], [], ls="none", marker="D", ms=10, color=SECOND)]
    labels += ["simulation (exact Cox process)", "σ = 3 simulation (biased low)",
               "theory beyond the simulated range", r"$\bar n^* = e^{\sigma^2}/\mathrm{CV}_\mathrm{rate}^2$"]
    _boxed_legend(ax, handles, labels, loc="upper left", bbox_to_anchor=(0.33, 0.995), fontsize=12.5, ncol=2,
                  columnspacing=1.2, handlelength=2.2)
    ax.text(2.5e5, 1.25, "σ = 3 (hollow): the variance comes from queries\n"
            "6 log-sd above the median (1 in 10⁹), which a finite\n"
            "simulation rarely draws; at 10¹⁰/day they happen daily.",
            fontsize=12.5, color=INK2, va="bottom", ha="left", bbox=LABEL_BOX)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5, 3e11)
    ax.set_ylim(0.3, 60)
    ax.set_xlabel("expected queries per day  n̄  (queries/day)")
    ax.set_ylabel("CV of daily energy  sd(S)/E[S]  (ratio)")
    ax.set_title(r"$\mathrm{CV}^2(S) = e^{\sigma^2}/\bar n + \mathrm{CV}_\mathrm{rate}^2$:   per-query noise "
                 r"falls like $1/\sqrt{\bar n}$, rate uncertainty does not", loc="left", fontsize=15, color=INK2, pad=10)
    fig.suptitle("At grid scale per-query noise averages out; what is left is uncertainty about the rate",
                 fontsize=18, fontweight="bold", x=0.5, y=0.975)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.85)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Delta Q distribution and quantile capacity (the handoff)
# ---------------------------------------------------------------------------


def fig4_handoff(dq_default, dq_alt, ratios, pjm_level: float = 30.0, label: str = "2030/31",
                 pjm_peak: float | None = None, alt_name: str = "compounding efficiency",
                 default_name: str = "efficiency matures (default)"):
    """CCDF of Delta Q on log axes with the newsvendor quantiles for each cost ratio marked."""
    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    x = np.geomspace(0.05, 500, 600)

    def ccdf(d):
        d = np.sort(np.asarray(d))
        return 1 - np.searchsorted(d, x, side="right") / d.size

    if pjm_peak:
        ax.axvspan(pjm_peak, 500, color=GRID, lw=0, zorder=0)
        ax.text(np.sqrt(pjm_peak * 500), 0.95, f"exceeds PJM's whole\n2026 summer peak (≈{pjm_peak:.0f} GW)",
                ha="center", va="top", rotation=90, fontsize=12.5, color=INK2)
    ax.plot(x, ccdf(dq_alt), color=MUTED, lw=2.6, label=alt_name, zorder=3)
    ax.plot(x, ccdf(dq_default), color=ACCENT, lw=3.4, label=default_name, zorder=4)
    for r in ratios:
        ax.axhline(r, color=AXIS, lw=1.3, zorder=1)
        ax.text(0.056, r * 1.06, rf"$c_\mathrm{{cap}}/c_\mathrm{{short}}$ = {r:.2f}", fontsize=13, color=INK2,
                va="bottom")
    qs = [float(np.quantile(dq_default, 1 - r)) for r in ratios]
    for r, q in zip(ratios, qs):  # up-right of a point is always above a falling CCDF
        ax.plot([q], [r], marker="o", ms=11, color=ACCENT, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{q:.1f} GW (P{round(100 * (1 - r))})", (q, r), xytext=(9, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=14, color=INK, fontweight="bold")
    mean = float(np.mean(dq_default))
    p_mean = float(np.mean(dq_default > mean))
    ax.plot([mean], [p_mean], marker="D", ms=10, color=INK, mec=SURFACE, mew=1.5, zorder=6)
    ax.annotate(f"mean {mean:.1f} GW: pulled up\nby the far tail", (mean, p_mean), xytext=(1.2, 0.03),
                textcoords="data", ha="left", va="center", fontsize=13, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    p30 = float(np.mean(np.asarray(dq_default) > pjm_level))
    ax.vlines(pjm_level, 1e-3, p30 * 1.45, color=INK, lw=2.2, zorder=2)  # stops just above the curve
    ax.plot([pjm_level], [p30], marker="o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=6)
    ax.text(pjm_level * 0.92, 0.0013, f"PJM: up to ~{pjm_level:.0f} GW of ALL\ndata-centre growth "
            f"2025–2030\n(P{100 * (1 - p30):.0f} of the default here)", fontsize=12.5, color=INK, va="bottom",
            ha="right")
    neg_d, neg_a = float(np.mean(np.asarray(dq_default) <= 0)), float(np.mean(np.asarray(dq_alt) <= 0))
    ax.text(0.056, 0.0105, f"P(ΔQ ≤ 0): {100 * neg_d:.1f}% (default), {100 * neg_a:.0f}% (compounding);\n"
            "those draws cannot appear on a log axis", fontsize=12.5, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 500)
    ax.set_ylim(1e-3, 1.05)
    ax.set_xlabel(f"new PJM summer-peak load from AI,  ΔQ  (GW, delivery year {label} minus 2025)")
    ax.set_ylabel("P(ΔQ > x)  (probability)")
    _boxed_legend(ax, loc="lower left", fontsize=13.5)
    med, q90 = float(np.quantile(dq_default, 0.5)), float(np.quantile(dq_default, 0.9))
    ax.set_title(r"Newsvendor reading (an analogy, not PJM's method): buy $Q$ with "
                 r"$P(\Delta Q > Q) = c_\mathrm{cap}/c_\mathrm{short}$", loc="left", fontsize=15, color=INK2, pad=10)
    fig.suptitle(f"Capacity for AI load is a high quantile: P90 = {q90:.0f} GW vs median {med:.1f} GW "
                 f"(DY {label})", fontsize=18, fontweight="bold", x=0.5, y=0.975)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.85)
    return fig
