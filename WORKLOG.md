# Work log (append-only)

Format: timestamp (local, EDT) — what — why — result — problems.

---

**2026-10-11 17:41** — Created project skeleton `ai-capacity-demo/` and copied `PROJECT_PROMPT.md` into it.
Why: spec §8 layout. Result: directories `refs/ src/aicap/ scripts/ notebooks/ tests/ outputs/{figures,cache,handoff}/ docs/`.

**2026-10-11 17:41** — Obtained the primary reference.
What: downloaded the PDF from the spec URL (business.columbia.edu …/Bassamboo-Randhawa-Zeevi-parameter-uncertainty.pdf) into a scratch folder and compared it with the copy already on disk (`~/Desktop/APMA4903/`).
Result: download succeeded; md5 `f2ede02434ff1af274cb197a2ce28f4c` for both, 37 pages, the working-paper version dated 10 May 2010. Copied to `refs/` (PDF is git-ignored).
Note: page numbers in our notes are this PDF's pages (they equal the printed page numbers); the journal version (Mgmt Sci 56(10):1668–1686) is paginated differently.

**2026-10-11 17:43** — Created `.venv` (Python 3.13.13, miniforge) and installed numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, jupyter, nbconvert 7.17.1, pytest 9.1.1, SALib 1.6.0, pypdf 6.20.0. Nothing installed globally.

**2026-10-11 17:44–17:52** — Read the paper (text extracted with pypdf). Findings that change or confirm the spec:
1. Notation: the paper's service rate is **θ** (spec: μ). Cost (eq. 1, p.7): Π(b) = (h + pγ)·E[N − b]⁺ + c·b with the expectation over both N and Λ — **identical to the spec's convention** C(b; λ) = c·b + (h + pγ)E_λ[Q], Π(b) = E_Λ[C(b; Λ)].
2. Method (p.10): steady state of the Markov chain conditional on Λ, then numerical integration over Λ; b* by evaluating many b and using convexity. Exact, not simulation (simulation is used only in §6.2).
3. §4 parameters (p.9): c = 1/3, p = h = 1, mean service time 1 (θ = 1), mean impatience 1/3 (γ = 3) → fractile 1/4, b̄ = 75th percentile.
4. Rounding: never stated (p.7 says b is treated as real), but all 20 reported b̄ values in Tables 2, 3, 4, 7 equal floor(F⁻¹(3/4)/θ) (43.75→43, 87.5→87, 37.5→37, 1.75→1, 162.5→162, 152.5→152).
5. Theorem 1 is on p.15 (statement copied verbatim into `refs/paper_notes.md`).

**2026-10-11 17:46–17:53** — Investigated the known reproduction discrepancy (scratch scripts, not in the repo).
- With the spec/paper convention, the prototype's numbers are reproduced exactly (U[25,50]: b* 45, Π* 17.10, b̄ 43, gap 0.41 %; λ = 37.5: 43, 15.66, 37, 7.3 %; etc.).
- **The same code reproduces the paper's Table 4 (p.13) on all six rows to ±0.02**: U[0,300] 224/88.33 vs 224/88.34; U[125,175] 165/59.06 = paper; U[135,165] 162/57.40 = paper; U[140,160] 162/56.80 vs 56.78; U[145,155] 161/56.40 = paper; λ = 150: b* 161, Π* 56.26, b̄ 150, Π(b̄) 58.25, gap 3.5 % — exactly the paper.
- Tables 2, 3 and 7 are **not** reproducible with any single parameter set. Tried: γ ∈ {0.25 … 6}, θ ∈ {0.9 … 1.05}, free (c, K) by least squares on 20 table entries; best RMS 0.17 (tables are given to 0.01). Backing out (c, K) row by row in Table 3 with γ = 3 gives c ≈ 1/3 every time but K = 4.86, 4.57, 4.00 (Table 4's λ = 150), 4.27 — not a constant, so no single cost weight explains Table 3.
- Table 7's small cases are far off (U[1,2]: paper Π(b̄ = 1) = 5.26; exact value 1.43, and the fluid lower bound is 1.00).
- Table 8 (single server, p.37) is internally inconsistent with its own stated parameters: c = 2/3, p = h = 1, γ = 1 give P(Λ > θ̄) = 1/3 (67th percentile, 41.67 for U[25,50]) but the table reports θ̄ = 33.33 (33rd percentile).
Conclusion: our implementation is the paper's model (Table 4 proves it); Tables 2, 3 and 7 seem to come from a different computation that the paper does not describe. Decision: use the paper's stated convention, report both sets of numbers side by side, add Table 4 as the exact-match validation. The qualitative result (≤ 1 % uncertain vs several % known) holds in both. Talk-text impact: "3–11 % off when the rate is known" is the paper's Table 3; our exact values are 2.6–7.3 %.

**2026-10-11 17:55** — git init, `.gitignore` (venv, PDF, regenerable handoff CSVs), `requirements.txt`, `requirements-lock.txt` (exact versions), `pyproject.toml` (editable install of `src/aicap` so scripts, tests and the notebook can `import aicap`).
Decision: the 2 × ~20 MB handoff CSVs are git-ignored (deterministic with the fixed seed; `run_all.py` rebuilds them); summary JSON and README in `outputs/handoff/` are committed.

**2026-10-11 17:56–18:02** *(time corrected, see 18:49 entry)* — Part A library: `src/aicap/erlang_a.py`, `src/aicap/capacity_rules.py`, tests.
- Stationary law: log π(n) = n·log λ − Σ_{k≤n} log d(k); the cumulative sum is shared by all arrival rates, so a batch of quadrature nodes is one outer product (fast). Truncation: d(n) is non-decreasing, so the mass beyond n_max is ≤ π(n_max)·r/(1−r) with r = λ/d(n_max+1); n_max grows until this bound and the matching E[Q]-tail bound are < 1e-12.
- Π(b) two ways: (i) "gauss": Gauss–Legendre for uniforms (spec), generalized Gauss–Laguerre (own Golub–Welsch, normalised weights) for gamma, Gauss–Hermite for lognormal; (ii) "split": Π(b) = Π̄(b) + (h+pγ)·E[D(Λ,b)], with Π̄ the newsvendor cost in closed form and D = E_λ[Q] − (λ−bθ)⁺/γ ≥ 0 (by flow balance, D = (θ/γ)·E[idle servers] above bθ). D is negligible more than ~12√b from bθ, so only that window needs composite Gauss–Legendre.
- Problem found and fixed: the first "split" version disagreed with Gauss–Legendre by ~2e-3 because D has a kink at λ = bθ (it subtracts (λ−bθ)⁺/γ) and the panels did not break there. With bθ as a panel edge the two methods agree to 1e-12.
- Finding: plain Gauss–Laguerre does NOT converge well for wide gamma rates (mean 1000, CV 0.3: 492.98 / 493.18 / 493.16 at 64/128/256 nodes) because E[Q|λ] bends sharply on the scale √E ≪ sd(Λ). The split method is stable to 1e-11 there. Decision: regime map (Part B) uses "split".
- Rules: b* by bisection on the forward difference Π(b+1)−Π(b) (valid for discretely convex Π; cross-checked against brute force on all 12 cases). The spec allowed integer golden-section; bisection is equivalent for convex sequences and simpler. b̄ rounded down (paper). b_det = argmin C(b; E[Λ]).
- Problem: a truncation test failed at 9e-11 absolute on E[Q] = 333 (2.8e-13 relative): floating-point summation noise between two summation orders, not truncation. Changed the comparison to relative.
- Result: 165 tests pass in 3.4 s, including exact reproduction of paper Table 4 (b*, b̄ exact; Π within 0.025) and the spec's prototype table.

**2026-10-11 18:02–18:15** *(time corrected)* — Figure 1, Part B grid, Part C module, Part D modules.
- Figure style: loaded the dataviz guidance; no Node.js on this machine, so I ported its palette validator to Python (scratch, not in repo) and confirmed the port reproduces the documented reference numbers (worst adjacent CVD ΔE 9.1, normal-vision 19.6). Validated sets: blue/orange categorical (all-pairs CVD ΔE 24.7), first three slots, blue ordinal ramp for σ = 1, 2, 3. Muted gray for comparison series; blue↔gray↔red diverging map.
- Figure 1 drafted and checked visually twice (fixed a legend gap and two label collisions).
- Part B: `regime_grid`/`boundary_analysis` in `capacity_rules.py`, `scripts/precompute_regime.py` → `outputs/cache/regime_grid.npz` (52 s total). Gamma grid 9 s, lognormal 10 s, refined-quadrature rerun 33 s. Convergence: refined vs default Π differ by < 1.4e-12 (relative), b* identical in all 225 cells. Result: the cheaper rule switches at CV ≈ κ/√E with κ = 1.40 (median over E; 1.30–1.90), i.e. parallel to the theorem's CV = 1/√E line (same −½ slope) but a factor 1.4 higher; 93.3 % of cells lie on the predicted side of CV = 1/√E itself and 99.6 % on the predicted side of the fitted line. Lognormal Λ: κ = 1.45, same cheaper rule in 99.1 % of cells.
- Part C: `cox.py` + 16 tests (z-tests at 0.1 % pass). Simulation for Figure 3 (σ = 1, 2, 3, CV_rate = 0.5, n̄ = 10…10⁶, ~2e8 draws per point) takes 44 s → cached. Finding: σ = 1 simulation within 3 % of theory; σ = 3 sample CV is far below theory at small n̄ (0.28× at n̄ = 10) because E[X²] is carried by queries 2σ = 6 log-sd above the median (1 in 10⁹). Not a bug: a heavy-tail effect; reported in the figure and guide.
- Part D: `params.py` (all priors with sources; presenter-1 block), `pjm_load.py`. 200k draws take 0.04 s. Matches the prototype within MC error: DY2030 default P10/P50/P90/mean = 0.946/5.435/34.12/16.76 vs 0.94/5.42/34.2/16.95; P(ΔQ > 30) = 0.1159 vs 0.116; P₀ median 0.496 vs 0.49; ladder ends at 4,759.5 MW vs 4,763.
- Problem: Fenton–Wilkinson lognormal check is off at the low quantiles of P_T (P10 −43 %, P50 −16 %, P90 +1 %). Cause: FW matches the mix sum's linear mean and variance, which the heavy-tailed agentic term (log-sd ≈ 1.36) dominates, so it overstates Var[ln mix] (FW var_log 2.34 vs MC 1.73). Added a second check with exact log-moments of the mix (4-D tensor Gauss quadrature): within 6 % at P10/P50/P90 and 0.3 % on mean/var of ln P_T. Both reported.
- Problem: quadrature nodes far in the tail of the (unbounded) lognormal w_A,T prior gave w_A > 1 and a negative mix (NaN log). Decision: cap w_A ≤ 1 − w_R in the model (shares must sum to ≤ 1). The cap binds with probability 2.1e-5 (≈ 4 of 200,000 draws); recorded as a deviation.

**2026-10-11 18:15–18:28** *(time corrected)* — Figures 2–4, `run_all.py`, handoff.
- Figures checked visually after each change. Fixes: colorbar label collided with the right panel (moved to an explicit 4-column gridspec), labels over data lines (moved), legend hiding the "AI at grid scale" label in Fig 3 (moved the label below the floor), Fig 4 quantile labels over the curve (placed up-right of each point, always above a falling CCDF) and the 30 GW line now stops just above its intersection with the curve.
- Font: Helvetica Neue lacked "→" (glyph warning). Switched to DejaVu Sans (ships with matplotlib, full glyph coverage); rendering now runs with UserWarnings as errors and is clean.
- Added a physical yardstick to Fig 4: PJM's forecast 2026 summer peak = 222,106 − 65,733 = 156,373 MW (both numbers from the PJM 2026 Load Forecast Report as cited in the spec). Finding: the ΔQ tail is unphysical beyond ~P98 (P99 = 182 GW, max 16,045 GW for DY2030 default; 1.3 % of draws exceed PJM's whole peak) and it drives the mean (16.8 GW). Flagged in the handoff README for presenter 3 (Jensen with a convex price curve will be dominated by these draws).
- Verified from the local PDF of Elsworth et al. (2508.15734v1.pdf, abstract): the 0.24 Wh median includes "AI accelerator power, host system energy, idle machine capacity, and data center energy overhead", and the 33x one-year reduction.
- `scripts/run_all.py` (8.6 s with caches, 53 s cold): Part A tables (Tables 2/3, 4, 7) + Fig 1; Part B from cache + Fig 2; Part C Cox simulation cache `outputs/cache/cox_sim.npz` + Fig 3; Part D summaries + Fig 4 + handoff (2 CSVs ≈ 16.6 MB each, JSON, README generated from the numbers); optional Sobol.
- Sobol (SALib, 2^14 base samples, 147,456 runs per structure, groups as in the spec): log-scale first-order indices match the closed form Var(group log-factor)/Var(ln P_T) to within 0.0005 (growth 0.313/0.314, mix 0.307, base volume 0.179, efficiency 0.097, Wh/query 0.047, training 0.027, siting 0.023/0.024 for 2030 default) — and the prototype's 0.31/0.31/0.17/0.09/0.05/0.03/0.02. The spec's linear-scale formula S_i = (e^{s_i²} − 1)/(e^{S²} − 1) only agrees roughly with SALib on P_T itself (e.g. mix 0.124 vs 0.151, base volume 0.024 vs 0.078): the group factors are not exactly lognormal and linear-scale Sobol estimates on this heavy-tailed output are noisy. Reported as approximate.
- Refactor: moved the Part D summary and handoff writer into `pjm_load.py` (`summarize`, `write_handoff`) so the notebook can call them.

**2026-10-11 18:28–18:33** *(time corrected)* — Notebook and end-to-end tests.
- `notebooks/demo.ipynb` (19 cells: setup + sections 1–6 of the storyboard), generated by `scripts/build_notebook.py` (kept so the cells stay reviewable as plain Python). Executed from a fresh kernel: 6.3 s total, slowest live cell 2.7 s (handoff export). Committed with outputs so it opens pre-run.
- Problem found while timing: one table cell failed inside pandas' Styler (`show(df, {})` → empty dict → numeric format applied to strings). IPython records display-formatter errors as outputs without failing the cell, so `nbconvert --execute` exits 0 anyway. Fixed `show()` (type-aware default format) and made the end-to-end test scan executed outputs for `error` entries, not just the exit code.
- Consistency: figure 2's AI boxes/labels are now built by one library function (`cox.regime_boxes`) for both run_all and the notebook; delivery-year CV via `pjm_load.cv_2030_range`.
- `AICAP_OUTPUTS` environment variable redirects all outputs; `tests/test_end_to_end.py` (marked slow) runs run_all.py and nbconvert against a temporary copy of the caches so the real outputs/ stays untouched: 2 passed in 18 s.
- Rehearsed the live toggle (EFFICIENCY = "compounding") on a copy: DY2030 median 0.98 GW, P90 12.95 GW; 30 GW at the 95.8 % quantile.

**2026-10-11 18:33–18:47** *(time corrected)* — Documentation and source checks.
- Checked three spec VERIFY items: Patterson et al. 2022 (arXiv 2204.05149, §7 and Conclusion: "about ⅗ of ML energy use is for inference and ⅖ for training" → τ ≈ 0.67, inside our prior; updated the τ source string); Halfin & Whitt 1981 (Operations Research 29(3):567–588 from the bibliographic record); Ross 12th-edition section numbers could not be confirmed (10th edition: §5.4.2 compound, §5.4.3 conditional Poisson) → stays VERIFY.
- `docs/GUIDE.md` is rendered from `docs/GUIDE_TEMPLATE.md` by `scripts/build_guide.py`: 222 placeholders resolved from `summary.json` (unresolved placeholders raise). Proofreading found and fixed: an unescaped "|" breaking two tables, standard errors printing as "± 0.00", "nan" for the one lognormal row without a switch, priors printed as "8e+09", hand-typed numbers in prose (replaced by placeholders), and a misquoted range from the paper's Table 1 (22.3–148.1%, not 26–148%).
- Added `test_jensen_gap_for_a_convex_price_curve` (the knowledge map cites it).
- `docs/REFERENCES.md`: every spec §12 source with status (checked this build / checked by author / VERIFY). `README.md` written.
- `run_info()` now reports `git_dirty` for code paths only (outputs written during the run are not "dirt").

**2026-10-11 18:49** — Correction and final clean run.
- Correction: the five entries above marked *(time corrected)* originally carried estimated clock times (up to "21:45") that were wrong. They now use the real times from the git commits (17:55:59, 18:01:53, 18:15:00, 18:28:24, 18:32:52, 18:46:38). Content unchanged.
- Final clean run at commit f8bfb71 (code tree clean): `run_all.py --recompute` rebuilt both caches and all outputs in 104 s; `summary.json` and the regime cache both record git f8bfb71, `git_dirty: False`. Notebook re-executed in place from a fresh kernel: 6.4 s total, slowest cell 2.7 s, no error outputs.
- Process note: one shell command silently did nothing (a mistyped script path piped into `tail` hid the error); caught by checking the output and re-run directly.
