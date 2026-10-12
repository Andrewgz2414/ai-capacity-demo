# AI load, parameter uncertainty and capacity sizing: code demo

Presenter 2's live demo for the APMA 4903 group talk *"What does AI really cost — in energy, and on your electric bill?"*
It turns presenter 1's energy per query (Wh) into distributions of AI electricity use (TWh/yr) and of new PJM summer-peak load ΔQ (GW). It then shows, following Bassamboo, Randhawa & Zeevi (2010), why the capacity to buy is a **quantile** of that distribution rather than its median. Finally it hands the energy samples to presenter 3, who turns them into water (operational water footprint, Li et al. 2025).

**Start here:** `docs/GUIDE.md`, the full guide (knowledge map, results, assumptions, talk script). Every number in it is rendered from `outputs/summary.json`.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # tested versions: requirements-lock.txt
.venv/bin/python -m pip install -e .                    # installs the `aicap` package from src/
.venv/bin/python -m pytest -q                           # unit + end-to-end tests
.venv/bin/python scripts/run_all.py                     # figures, summary.json, handoff (seconds with caches)
.venv/bin/jupyter lab notebooks/demo.ipynb              # the live demo (~13 min: derivation, demo, pseudocode)
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
  pjm_load.py         ΔQ and energy sampler, quantiles with bootstrap SEs, Fermi ladder, lognormal checks, handoff
  plotting.py         projector style and the five figures
scripts/              precompute_regime.py, run_all.py, build_guide.py, build_notebook.py
notebooks/demo.ipynb  the talk, pre-run: Part I derivation, demo cells 1–5, Part III pseudocode, bridge cell 6
tests/                unit tests and end-to-end tests (marked slow)
outputs/figures/      fig0_newsvendor_foc, fig1_reproduction, fig2_regime_map, fig3_cox_floor, fig4_handoff (PNG 200 dpi + SVG)
outputs/cache/        regime_grid.npz, cox_sim.npz
outputs/handoff/      energy_samples_2028.csv, energy_samples_2030.csv  (TWh/yr; for presenter 3, water)
                      dQ_samples_DY2028.csv, dQ_samples_DY2030.csv      (GW of PJM peak; for price users)
                      handoff_summary.json, README.md
outputs/summary.json  every number quoted in the docs
docs/                 GUIDE.md (rendered), GUIDE_TEMPLATE.md, REFERENCES.md
refs/                 paper_notes.md (the paper PDF is not committed; link inside)
WORKLOG.md, TASKS.md  how the project was built
```

The handoff CSVs (17–26 MB each) are not committed; `run_all.py` regenerates them byte-for-byte from the fixed seed. Setting `AICAP_OUTPUTS=/some/dir` redirects all outputs; the end-to-end tests use this.

## How the talk runs

1. **Derivation (Part I).** The newsvendor first-order condition, c_cap = c_short · P(ΔQ > Q*), makes capacity a quantile of demand. A brute-force check on the simulated draws confirms it (Fig. 0).
2. **Demo (Part II).** When the quantile rule is safe (the paper, the regime map, the Cox process), and what it gives for AI load in PJM.
3. **Pseudocode (Part III).** The whole method in eight lines, mapped to the code and rerun with fresh random numbers.

The guide's §8 has the script for each part.

## Headline results (see the guide for the numbers with their errors)

- **Paper reproduction.** The paper's Table 4 is reproduced exactly. Tables 2, 3 and 7 differ slightly; the investigation is in the guide §6. The pattern holds: the newsvendor rule is within 1% when the rate is uncertain and several percent off when it is known.
- **Regime map.** The cheaper rule switches at CV ≈ 1.4/√E, parallel to Theorem 1's CV = 1/√E.
- **Cox process.** At ~10¹⁰ queries/day per-query noise is negligible, and AI sits far inside the uncertainty-dominated regime.
- **New PJM peak load from AI.** The 2030/31 distribution is heavily right-skewed: the median is a few GW and P90 is tens of GW. Capacity read as a quantile is therefore far above the median.
- **Bridge to presenter 3 (water).** The same draws give annualized electricity use for the world, the US and PJM. Water is linear in energy, but whether a water budget holds depends on the whole distribution, which is the same quantile logic. The handoff README states the energy boundary (approximately facility energy) and how it enters Li et al.'s formula.
