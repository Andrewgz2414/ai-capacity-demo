"""Shared figure style and the four demo figures.

Projector style: fonts >= 14 pt, data lines >= 2.5 pt, solid hairline grid, light surface.
Colours come from a validated colour-blind-safe reference palette (blue / orange categorical
slots, a one-hue blue ramp for ordered series, blue <-> red diverging with a gray midpoint);
one accent colour (blue) marks the point of each plot and comparisons are gray.
Every figure is saved as PNG (200 dpi) and SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette (light surface)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # primary text
INK2 = "#52514e"  # secondary text
MUTED = "#898781"  # axis labels, de-emphasised series
GRID = "#e1e0d9"  # hairline grid
AXIS = "#c3c2b7"  # baseline / spines
ACCENT = "#2a78d6"  # categorical slot 1 (blue): the point of the plot
SECOND = "#eb6834"  # categorical slot 2 (orange)
BLUE_RAMP = {250: "#86b6ef", 450: "#2a78d6", 650: "#104281", 100: "#cde2fb", 350: "#5598e7", 550: "#1c5cab"}
NEUTRAL_MID = "#f0efec"
RED = "#e34948"

# Diverging map: blue (negative) <-> gray midpoint <-> red (positive). Equal arms.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#86b6ef", NEUTRAL_MID, "#f2a7a3", RED, "#a3282a"], N=256)

FIG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
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
    outdir = Path(outdir) if outdir else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{name}.png", outdir / f"{name}.svg"]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    return paths


def _pct(x: float) -> str:
    if abs(x) < 5e-5:
        return "0.00%"
    return f"{100 * x:.2f}%" if abs(x) < 0.1 else f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Figure 1: reproduction of the paper's section 4
# ---------------------------------------------------------------------------


def fig1_reproduction(rows: list[dict]):
    """rows: reproduction rows for U[25,50], U[50,100], U[200,400] and lambda = 37.5, 75, 300."""
    apply_style()
    random_rows = sorted([r for r in rows if r["cv"] > 0], key=lambda r: r["E"])
    fixed_rows = sorted([r for r in rows if r["cv"] == 0], key=lambda r: r["E"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    panels = [(axes[0], random_rows, "Rate uncertain: Λ ~ U[a, 2a]  (CV 19%)"),
              (axes[1], fixed_rows, "Rate known: λ fixed")]
    for ax, rr, title in panels:
        E = np.array([r["E"] for r in rr])
        nv = 100 * np.array([r["gap_bar"] for r in rr])
        kr = 100 * np.array([r["gap_det"] for r in rr])
        paper = 100 * np.array([r["paper_gap_bar"] for r in rr])
        ax.plot(E, kr, color=MUTED, marker="s", lw=2.5, ms=9, label="known-rate rule  b = argmin C(b; E[Λ])",
                zorder=3)
        ax.plot(E, nv, color=ACCENT, marker="o", lw=3.2, ms=10, label="newsvendor rule  b̄ = 75th pct of Λ",
                zorder=4)
        ax.plot(E, paper, ls="none", marker="o", ms=15, mfc="none", mec=ACCENT, mew=1.8,
                label="newsvendor, paper's Tables 2–3", zorder=5)
        below = True  # labels under the points keep clear of the paper markers above
        for x, y in zip(E, nv):
            ax.annotate(_pct(y / 100), (x, y), xytext=(0, -14 if below else 12), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", fontsize=14, color=INK,
                        fontweight="bold")
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
               bbox_to_anchor=(0.5, 0.875), fontsize=13.5, handlelength=2.2, columnspacing=1.6)
    nv_unc = max(r["gap_bar"] for r in random_rows)
    nv_det = [r["gap_bar"] for r in fixed_rows]
    fig.suptitle(f"When the rate is uncertain the newsvendor rule is within {_pct(nv_unc)} of optimal; "
                 f"when it is known it is {_pct(min(nv_det))}–{_pct(max(nv_det))} off",
                 x=0.5, y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.76, wspace=0.08)
    return fig
