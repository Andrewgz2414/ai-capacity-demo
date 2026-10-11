# Task list

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (reason given)

## 0. Setup
- [x] Project skeleton, venv (`.venv`), `requirements.txt`, `pyproject.toml` (editable install of `src/aicap`)
- [x] Paper PDF into `refs/` (downloaded from the spec URL; byte-identical to the local copy, md5 f2ede024…)
- [x] Read §3, §4, §5 (and §6, appendices); diagnose the reproduction discrepancy
- [x] `refs/paper_notes.md` with page numbers
- [x] git init, `WORKLOG.md`, this file

## Part A — reproduce the paper (spec §3)
- [x] `erlang_a.py`: log-space stationary law, adaptive truncation (tail < 1e-12), E[Q], C(b; λ), Π(b) for point / uniform / gamma / lognormal rates
- [x] `capacity_rules.py`: optimal b* (brute force + discrete-convexity check, then bisection on the forward difference), newsvendor b̄ (floor, as in the paper), known-rate b_det, fitted β, gaps
- [x] Tests: M/M/∞ (γ = μ), b = 0, flow balance, monotone E[Q], truncation, unimodality, newsvendor 43/87/350, fixed-λ newsvendor = λ, quadrature convergence
- [x] Reproduction table (six §4 cases) + extended validation against Tables 4 and 7
- [x] Figure 1 (`fig1_reproduction`)

## Part B — regime map (spec §4)
- [x] `scripts/precompute_regime.py`: 15 × 15 grid (E 10…3000, CV 0.01…0.6), gamma Λ; lognormal robustness; quadrature convergence check; cache `outputs/cache/regime_grid.npz`
- [x] Figure 2 (`fig2_regime_map`) with CV = 1/√E, the paper's cases, AI inset

## Part C — Cox process (spec §5)
- [x] `cox.py`: lognormal moments, compound-Poisson simulation, theory CV, crossover n̄*, E_eff, concurrency R, Rao–Blackwell helper
- [x] Tests: simulated mean/variance z-tests, lognormal moments
- [x] Figure 3 (`fig3_cox_floor`); AI regime numbers

## Part D — PJM ΔQ and quantile capacity (spec §6)
- [x] `params.py` (all priors, ranges, sources; presenter-1 block in one place)
- [x] `pjm_load.py`: sampler, P0, PT, ΔQ, both efficiency structures, quantiles + bootstrap SEs, Fenton–Wilkinson check, Fermi ladder, quantile capacity, cost of uncertainty, flexible load
- [x] Tests: ladder vs prototype (2 %), seed stability, units (35.4 MW)
- [x] Figure 4 (`fig4_handoff`); handoff CSVs, JSON, README
- [x] Optional: Sobol (SALib) on ln P_T + closed-form check

## Integration
- [x] `plotting.py` shared style (≥14 pt, ≥2 pt lines, colour-blind-safe, PNG 200 dpi + SVG)
- [x] `scripts/run_all.py` → `outputs/summary.json` + figures + handoff
- [x] `notebooks/demo.ipynb` (cells 0–6, < 2 min fresh, < 5 s per live cell)
- [x] End-to-end: run_all, nbconvert, visual check of every figure

## Documentation (spec §10)
- [x] `docs/GUIDE.md` (rendered from a template so every number comes from `summary.json`)
- [x] `docs/REFERENCES.md`
- [x] `README.md`
- [x] Final report in chat (after the final clean run)
