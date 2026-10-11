# AI load, parameter uncertainty and capacity sizing: code demo

Presenter 2's live demo for the APMA 4903 group talk *"What does AI really cost — in energy, and on your electric bill?"*
It turns presenter 1's energy per query (Wh) into a distribution of new PJM summer-peak load ΔQ (GW). It then shows, following Bassamboo, Randhawa & Zeevi (2010), why the capacity to buy is a **quantile** of that distribution rather than its median, and hands the samples to presenter 3.

**Start here:** `docs/GUIDE.md`, the full guide (knowledge map, results, assumptions, talk script). Every number in it is rendered from `outputs/summary.json`.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # tested versions: requirements-lock.txt
.venv/bin/python -m pip install -e .                    # installs the `aicap` package from src/
.venv/bin/python -m pytest -q                           # unit + end-to-end tests
.venv/bin/python scripts/run_all.py                     # figures, summary.json, handoff (seconds with caches)
.venv/bin/jupyter lab notebooks/demo.ipynb              # the live demo (~12 min storyboard)
```

`scripts/run_all.py --recompute` rebuilds the two caches (regime map and Cox simulation), which takes minutes. `scripts/build_guide.py` re-renders `docs/GUIDE.md` from `outputs/summary.json`. `scripts/build_notebook.py` regenerates the notebook from plain Python.

**Presenter 1's numbers** (median Wh per chat query, per-query log-sd, reasoning and agentic multipliers) are placeholders in one block, `PRESENTER1` in `src/aicap/params.py`. Edit it, rerun `run_all.py` and `build_guide.py`.

## What is where

```
src/aicap/
  erlang_a.py         exact M/M/b+M stationary law, E[Q], cost C(b; λ), Π(b) for random rates
  capacity_rules.py   optimal, newsvendor and known-rate rules; paper tables; regime grid
  cox.py              compound/Cox Poisson model, simulation, regime numbers for AI
  params.py           every prior with range and source (presenter-1 block at the top)
  pjm_load.py         ΔQ sampler, quantiles with bootstrap SEs, Fermi ladder, lognormal checks, handoff
  plotting.py         projector style and the four figures
scripts/              precompute_regime.py, run_all.py, build_guide.py, build_notebook.py
notebooks/demo.ipynb  the demo (cells 0–6), pre-run
tests/                unit tests and end-to-end tests (marked slow)
outputs/figures/      fig1_reproduction, fig2_regime_map, fig3_cox_floor, fig4_handoff (PNG 200 dpi + SVG)
outputs/cache/        regime_grid.npz, cox_sim.npz
outputs/handoff/      dQ_samples_DY2028.csv, dQ_samples_DY2030.csv, handoff_summary.json, README.md
outputs/summary.json  every number quoted in the docs
docs/                 GUIDE.md (rendered), GUIDE_TEMPLATE.md, REFERENCES.md
refs/                 paper_notes.md (the paper PDF is not committed; link inside)
WORKLOG.md, TASKS.md  how the project was built
```

The handoff CSVs (~17 MB each) are not committed; `run_all.py` regenerates them byte-for-byte from the fixed seed. Setting `AICAP_OUTPUTS=/some/dir` redirects all outputs; the end-to-end tests use this.

## Headline results (see the guide for the numbers with their errors)

- **Paper reproduction.** The paper's Table 4 is reproduced exactly. Tables 2, 3 and 7 differ slightly; the investigation is in the guide §6. The pattern holds: the newsvendor rule is within 1% when the rate is uncertain and several percent off when it is known.
- **Regime map.** The cheaper rule switches at CV ≈ 1.4/√E, parallel to Theorem 1's CV = 1/√E.
- **Cox process.** At ~10¹⁰ queries/day per-query noise is negligible, and AI sits far inside the uncertainty-dominated regime.
- **New PJM peak load from AI.** The 2030/31 distribution is heavily right-skewed: the median is a few GW and P90 is tens of GW. Capacity read as a quantile is therefore far above the median.
