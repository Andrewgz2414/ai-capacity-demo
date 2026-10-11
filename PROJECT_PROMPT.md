# Project prompt: AI load, parameter uncertainty and capacity sizing (code demo)

You are building a complete, runnable Python project for a live 10–15 minute code demo in a university talk. Read this whole file before writing any code. Then make a task list and work through it, verifying as you go. Everything you need is specified below; where something is uncertain it is marked **VERIFY** or **ASSUMPTION**.

---

## 0. How to work

1. **Plan first.** Write a task list covering sections 3–10. Keep it updated.
2. **Get the paper before coding Part A.** Download the primary reference PDF (section 2) to `refs/`. If the download fails, stop and ask me to put the PDF in `refs/`. Read Sections 3, 4 and 5 and write `refs/paper_notes.md`: the exact model, cost function, newsvendor prescription, the exact statement of Theorem 1, the numerical setup and every number in the Section 4 tables, each with its page number.
3. **Build bottom-up:** library code with tests → precompute scripts → notebook → documentation.
4. **Never invent numbers or citations.** Every number in the docs must come from a run of this code (via `outputs/summary.json`) or from a cited source. Label assumptions as assumptions and analogies as analogies.
5. **Keep a work log.** Append to `WORKLOG.md` as you go: timestamp, what you did, why, the result, and any problem.
6. **Use git.** Initialize a repo and commit at each milestone with a clear message.
7. **Ask at most three questions, and only if blocked.** Otherwise make the most reasonable choice, record it in `WORKLOG.md`, and continue.
8. **Finish with a short report in chat:** what was built, the key numbers, every deviation from this spec, and anything left unresolved.

---

## 1. Context

- **Course and talk:** Columbia APMA 4903 group talk, "What does AI really cost — in energy, and on your electric bill?" Four presenters; I am presenter 2. My segment is a live Jupyter code demo, 10–15 minutes.
- **Presenter 1** derives the energy of one AI query (Wh). Per query type k (chat, reasoning, agentic) the energy is approximately lognormal with median M_k and log-sd σ_k. Only the mean, M_k·e^(σ_k²/2), survives aggregation.
- **Presenter 3** models PJM's capacity market: how the clearing price responds to new load ΔQ (GW). The price is convex where supply is steep.
- **My job:** turn presenter 1's Wh into a distribution of new summer-peak load ΔQ (GW) in PJM, explain why the capacity bought should be a quantile of that distribution rather than its median, and hand the samples to presenter 3.

**My nugget (the talk text, for context):**

> My primary reference is "Capacity Sizing Under Parameter Uncertainty: Safety Staffing Principles Revisited" by Bassamboo et al. (*Management Science*, 2010). My nugget is a code demo showing that when the demand rate itself is uncertain, the best capacity is a quantile of that uncertainty (a newsvendor solution), not the mean plus a square-root safety margin. This is in Sections 3 and 4 of the paper. I'll code the paper's queueing model and show that it reproduces the Section 4 results: the newsvendor rule is within 1% of optimal when the rate is uncertain, but 3–11% off when the rate is known. I'll then apply it to AI. Queries reaching data centers are the paper's arrivals, and presenter 1's per-query energy sets their noise. At ~10¹⁰ queries a day that noise averages out, while uncertainty about 2030 volume and query mix spans an order of magnitude. So the GW I hand to presenter 3 is a distribution of new PJM peak load, and the capacity bought should be a high quantile of it, not the median. That gap between quantile and median is where AI's uncertainty itself shows up on electric bills.

---

## 2. Primary reference

A. Bassamboo, R. S. Randhawa, A. Zeevi (2010). "Capacity sizing under parameter uncertainty: safety staffing principles revisited." *Management Science* 56(10):1668–1686. doi:10.1287/mnsc.1100.1203.
- PDF: https://business.columbia.edu/sites/default/files-efs/pubfiles/5051/Bassamboo-Randhawa-Zeevi-parameter-uncertainty.pdf
- Record: https://ideas.repec.org/a/inm/ormnsc/v56y2010i10p1668-1686.html
- Structure: §3 model and newsvendor prescription; §4 numerical illustration; §5 theory (Theorem 1); §6 robustness.

**What I believe the paper says (VERIFY every item against the PDF; my notes came from an automated reader):**

- Model: M/M/b+M queue (Erlang-A). Arrival rate Λ is random with distribution F and mean λ̄; given Λ, arrivals are Poisson. b servers with service rate μ; customers abandon at rate γ while waiting.
- Costs: staffing c per server per unit time; holding h per waiting customer per unit time; penalty p per abandonment.
- Newsvendor (fluid) problem: minimize (p + h/γ)·E[(Λ − μb)⁺] + c·b, so P(Λ > μb̄) = c / (μ(p + h/γ)).
- Regime test: compare CV_λ = sd(Λ)/E[Λ] with 1/√E, where E = λ̄/μ is the offered load in Erlangs. When CV_λ ≫ 1/√E, rate uncertainty dominates and the newsvendor rule is near-optimal; when CV_λ ≪ 1/√E, Poisson variability dominates and square-root safety staffing is needed. State Theorem 1 exactly as the paper does.
- Section 4 setup: μ = 1, γ = 3, c = 1/3, p = 1, h = 1, so the critical fractile is P(Λ > μb̄) = 1/4 and b̄ is the 75th percentile of Λ. Cases: Λ ~ U[25, 50], U[50, 100], U[200, 400] (CV ≈ 19.2%), and fixed λ = 37.5, 75, 300.
- Section 4 results as extracted (VERIFY):

| Case | b* | Π* | b̄ | Π(b̄) | Gap |
| --- | --- | --- | --- | --- | --- |
| U[25, 50] | 46 | 17.34 | 43 | 17.49 | 0.9% |
| U[50, 100] | 89 | 33.13 | 87 | 33.20 | 0.2% |
| U[200, 400] | 351 | 127.08 | 350 | 127.08 | 0.006% |
| λ = 37.5 | 42 | 15.94 | 37 | 17.67 | 10.9% |
| λ = 75 | 83 | 29.64 | 75 | 31.60 | 6.6% |
| λ = 300 | 316 | 109.01 | 300 | 112.4 | 3.1% |

The reported b̄ (43, 87, 350) is consistent with rounding the 75th percentile down; confirm the paper's rounding rule.

---

## 3. Part A — Reproduce the paper (module `erlang_a.py`, `capacity_rules.py`)

**Exact model.** Given Λ = λ and b servers, the number in system N is a birth–death chain with birth rate λ and death rate d(n) = min(n, b)·μ + (n − b)⁺·γ. Stationary law:

π(n) ∝ ∏_{k=1}^{n} λ / d(k).

Compute it in log space and truncate where the remaining tail mass is below 1e−12 (check this numerically). Queue length Q = (N − b)⁺; abandonment rate = γ·E[Q].

**Cost.** For known λ: C(b; λ) = c·b + (h + p·γ)·E_λ[Q]. With random Λ, capacity is chosen before Λ is revealed: Π(b) = E_Λ[C(b; Λ)]. Integrate over Λ with Gauss–Legendre quadrature for uniforms (check convergence in the number of nodes). **VERIFY this convention against §3 of the paper** and use the paper's convention if it differs.

**Three capacity rules.**
1. **Optimal b\*:** integer brute force (the cost is unimodal in b; justify or check it, then you may use an integer golden-section search for speed).
2. **Newsvendor b̄:** ignores Poisson variability. b̄ = F⁻¹(1 − c/(μ(p + h/γ)))/μ, rounded as in the paper.
3. **Known-rate rule b_det:** ignores rate uncertainty. The optimal capacity if Λ were exactly its mean: argmin_b C(b; E[Λ]). Also report the fitted safety factor β in b_det ≈ E + β√E (square-root staffing).

Gap of a rule = Π(rule)/Π* − 1.

**Reproduction table.** For the six Section 4 cases, report b\*, Π\*, b̄, Π(b̄), gap(b̄), b_det, gap(b_det), side by side with the paper's numbers.

**Known discrepancy to resolve.** A prototype using exactly the convention above got the same pattern but not the same numbers:

| Case | b* | Π* | b̄ | gap(b̄) |
| --- | --- | --- | --- | --- |
| U[25, 50] | 45 | 17.10 | 43 | 0.41% |
| U[50, 100] | 88 | 32.92 | 87 | 0.05% |
| U[200, 400] | 350 | 126.90 | 350 | 0.00% |
| λ = 37.5 | 43 | 15.66 | 37 | 7.3% |
| λ = 75 | 83 | 29.45 | 75 | 4.7% |
| λ = 300 | 315 | 108.83 | 300 | 2.6% |

Variants tried that did not match: holding cost on all customers in system; γ = 1/3; γ = 1. Read §3–4 to find the paper's exact cost definition and computation method (exact, approximation, or simulation), try to match, and document the outcome in `refs/paper_notes.md` and the guide. If you cannot match exactly, report both sets of numbers honestly. The qualitative result is what the talk needs.

**Plot 1 (`fig1_reproduction`).** Optimality gap (%) of the newsvendor rule and the known-rate rule against mean load (37.5, 75, 300), in two panels: random rate (left) and fixed rate (right). Title states the finding.

---

## 4. Part B — Regime map (`scripts/precompute_regime.py`)

Over a grid of offered load E (log-spaced, ~10 to ~3,000, about 15 values) and rate CV (log-spaced, ~0.01 to ~0.6, about 15 values), compute Π\*, Π(b̄) and Π(b_det) with the Section 4 cost parameters. Use a gamma distribution for Λ with the given mean and CV (generalized Gauss–Laguerre quadrature, or a fine grid; check convergence). Also run one alternative distribution (lognormal) as a robustness check, in the spirit of the paper's §6.

Cache the results to `outputs/cache/regime_grid.npz`; the notebook loads the cache by default and recomputes only when `RECOMPUTE = True`.

**Plot 2 (`fig2_regime_map`).** A map of which rule is cheaper and by how much (for example log10 of gap(b̄)/gap(b_det) on a diverging scale), with the line CV = 1/√E overlaid and the paper's three cases marked. If the line does not separate the regions as the theorem predicts, say so honestly. Add an arrow or inset showing that AI demand at grid scale lies far beyond the right edge (Part C).

---

## 5. Part C — AI queries as a Cox process (module `cox.py`)

**Model.** Daily query count N | ξ ~ Poisson(n̄·ξ) with E[ξ] = 1 and CV(ξ) = CV_rate; energies X_i i.i.d. lognormal with median M and log-sd σ; S = Σ_{i≤N} X_i. Moments: m₁ = M·e^(σ²/2), m₂ = M²·e^(2σ²).

**Key results to implement and test:**
- Compound Poisson: E[S | ξ] = n̄ξ·m₁ and Var(S | ξ) = n̄ξ·m₂ (Wald's identity plus the Poisson variance).
- Law of total variance: Var S = E[Λ·m₂] + Var(Λ·m₁), the aleatory plus the epistemic part.
- With only the rate uncertain: CV²(S) = e^(σ²)/n̄ + CV_rate². The first term vanishes like 1/n̄; the second is a floor.
- Crossover: n̄* = e^(σ²)/CV_rate².
- Mapping onto the paper's axis: effective load E_eff = n̄/e^(σ²), so the per-query noise is 1/√E_eff.
- Rao–Blackwellization: at grid scale replace S by E[S | θ] = Λ(θ)·m₁(θ). The mean is unchanged and the variance can only fall, so the grid model needs no inner loop.

**Plot 3 (`fig3_cox_floor`).** Log-log CV of daily energy against n̄ (10 to 10⁶ simulated; beyond that the theory line, dashed) for σ = 1, 2, 3 with CV_rate = 0.5 (ASSUMPTION; make it a parameter). Overlay the theory curve and the epistemic floor, and mark (a) a typical benchmark size n = 100 to 1,000 and (b) AI at grid scale, n̄ ≈ 10¹⁰ queries/day.

**Placing AI on the regime map.** Report two numbers with their assumptions:
- Daily-energy scale: E_eff = n̄/e^(σ²) with n̄ = 10¹⁰ to 2×10¹⁰ per day and σ = 1 to 3.
- Instantaneous concurrency: R = (arrivals per second) × E[service time]. ChatGPT's 2.5×10⁹ prompts/day is ≈ 2.9×10⁴ per second; ASSUMPTION E[S] = 10 s (range 2–60 s) gives R ≈ 3×10⁵ and 1/√R ≈ 0.002.

Compare these with CV_rate for day-ahead forecasts (ASSUMPTION 0.05–0.2) and for 2030 (from Part D, order 1). Conclude which regime AI is in.

---

## 6. Part D — PJM new peak load ΔQ and quantile capacity (module `pjm_load.py`, `params.py`)

**Model.** Draw independent epistemic parameters θ (table below). With horizon T years from 2025 (summer 2028 for delivery year 2028/29, T = 3; summer 2030 for DY 2030/31, T = 5; PJM delivery years run June to May):

- m_c = M_c · e^(σ_c²/2) · β  (mean Wh per chat query, including fleet overhead)
- mix_t = (1 − w_R,t − w_A,t) + w_R,t · r + w_A,t · a  (energy per request relative to chat)
- G_T = exp( ln(g₁)/λ · (1 − e^(−λT)) )  (volume growth, Gompertz)
- H_T = exp( ln(h)/λ · (1 − e^(−λT)) ) by default (efficiency matures with adoption); toggle `EFFICIENCY = "compounding"` gives H_T = h^T
- scope = (1 + τ) · s_US · s_PJM · κ
- P₀ = N₀ · m_c · mix₀ · scope / 24 / 10⁹  [GW]  (Wh/day ÷ 24 h = W)
- P_T = N₀ · G_T · m_c · mix_T / H_T · scope / 24 / 10⁹  [GW]
- ΔQ = P_T − P₀ (may be negative; keep the sign)

Keep all inputs in `params.py`, each with its distribution, range and source. Presenter 1's numbers (M_c, σ_c, r, a) are placeholders until they hand over theirs; make them a single obvious place to edit.

Lognormal priors are given as P5–P95: μ = (ln lo + ln hi)/2 and s = (ln hi − ln lo)/(2 × 1.6449).

| Parameter | Symbol | Distribution | Range | Source |
| --- | --- | --- | --- | --- |
| Global AI requests/day, 2025 | N₀ | lognormal | 8×10⁹ – 5×10¹⁰ | ChatGPT 2.5 B/day (July 2025); Google 480 T tokens/month (May 2025) at an ASSUMED 1–3 k tokens/request; other providers by judgment |
| Median Wh per chat query | M_c | lognormal | 0.20 – 0.40 | Google 0.24 Wh (Elsworth et al. 2025); Altman 0.34 Wh; Oviedo et al. 0.31 Wh |
| Per-query log-sd | σ_c | uniform | 0.7 – 1.3 | Oviedo IQR 0.16–0.60 around 0.31 implies σ ≈ 0.98; ±0.3 by judgment |
| Fleet overhead | β | uniform | 1.00 – 1.31 | Oviedo β = 1.31 = 1.10 network × 1.08 daily load × 1.10 redundancy; 1.0 if the median already includes idle capacity |
| Reasoning energy multiplier | r | lognormal | 5 – 30 | Oviedo: 13× median energy for 15× tokens; AI Energy Score v2: 30× average (GPU-only, non-production) |
| Agentic energy multiplier per task | a | lognormal | 30 – 1000 | Bai et al. 2026: ~1000× tokens, input-dominated, so energy multiple lower. ASSUMPTION |
| Reasoning share 2025 / delivery year | w_R0 / w_RT | uniform | 0.05–0.15 / 0.15–0.50 | Oviedo's 90/10 split; later share by judgment |
| Agentic share 2025 / delivery year | w_A0 / w_AT | uniform / lognormal | 0.001–0.005 / 0.005–0.08 | ASSUMPTION; widest prior |
| First-year volume growth | g₁ | lognormal | 1.8 – 5.0 | ChatGPT 1 B → 2.5 B/day (Dec 2024–Jul 2025); Google tokens 6.7× (May 2025–May 2026) |
| Growth decay rate | λ | uniform | 0.2 – 0.7 /yr | Two-point fit to Google's token growth: ln(49) → ln(6.7) gives ≈ 0.7 |
| First-year efficiency gain | h | lognormal | 1.15 – 2.2 | Oviedo 8–20× line-of-sight; Blackwell 1.5–2.5× per watt; AI Energy Score v2: most newer models not more efficient |
| Training-to-inference energy ratio | τ | lognormal | 0.25 – 1.5 | Patterson et al. 2022 (~3/5 inference). VERIFY |
| US share of global AI load | s_US | uniform | 0.40 – 0.60 | IEA 2025: US 45% of 2024 data-centre electricity |
| PJM share of US data-centre load | s_PJM | uniform | 0.30 – 0.50 | PJM Nov 2025: PJM ≈ 40% of US data centres in 2025 |
| Summer-peak coincidence factor | κ | uniform | 1.0 – 1.3 | Oviedo S5: peak/average node power 1/(f + (1−f)/(1+a)) = 1.235; PJM's flat 70% utilization ≈ 1.0 |

**Outputs.**
- 200,000 draws per delivery year (fixed seed), for both efficiency structures.
- P10, P50, P90, mean and P(ΔQ > 30 GW), with Monte Carlo standard errors (bootstrap for quantiles).
- Check the Monte Carlo against a lognormal approximation built from the log-variance (Fenton–Wilkinson for the mix sum); report the agreement.
- A "Fermi ladder": the median path from 35 MW (2.5×10⁹ prompts/day × 0.34 Wh ÷ 24 h) to the PJM number, one multiplicative factor per step.

**Quantile capacity (the newsvendor reading; label it an illustrative analogy, not PJM's method).** If a planner buys capacity Q with P(ΔQ > Q) = c_cap/c_short, then Q is a quantile of ΔQ.
- Report Q for cost ratios 0.05, 0.10, 0.25 and 0.50.
- Report the quantile at which PJM's ~30 GW sits and the implied cost ratio. Note that PJM's ~30 GW covers all data-centre growth 2025–2030, not AI alone.
- **Cost of uncertainty:** Q_0.9 − median. Show how it shrinks if the uncertainty in growth and mix is halved (halve their log-sds): the "better forecasting" policy hook.
- **Flexible load:** a lower c_short (curtailable data centres) lowers the fractile and the capacity needed. One sentence and one number.

**Plot 4 (`fig4_handoff`).** Distribution of ΔQ for DY 2030/31 (histogram or CCDF on a log axis), with median, mean, P90 and the newsvendor quantiles for the cost ratios marked, PJM's ~30 GW line, and both efficiency structures overlaid. Title states the finding.

**Handoff to presenter 3.**
- `outputs/handoff/dQ_samples_DY2028.csv` and `dQ_samples_DY2030.csv`: columns `draw_id, dQ_GW, P0_GW, PT_GW, efficiency_structure`. Put a header comment with the seed, number of draws, priors version and date.
- `outputs/handoff/handoff_summary.json` with the quantile tables.
- A short `outputs/handoff/README.md` explaining the units, what ΔQ includes, and that presenter 3 should push the samples through their price function directly (Jensen: E[p(ΔQ)] ≥ p(E[ΔQ]) for convex p).

**Optional, do last and only if time permits:** variance-based sensitivity (Sobol, SALib) on ln P_T with parameter groups (growth, mix, base volume, efficiency, Wh per query, training, siting and peak), plus a check against the closed form S_i = (e^(s_i²) − 1)/(e^(S²) − 1) for a product of independent lognormals.

---

## 7. Notebook storyboard (`notebooks/demo.ipynb`, ~12 minutes)

| Cell | Minutes | Content |
| --- | --- | --- |
| 0 | — | Setup (imports, seed, `RECOMPUTE = False`, plotting style). Pre-run before the talk. |
| 1 | 1 | Inputs from presenter 1: median, σ, mean per query type; one line showing mean = median·e^(σ²/2). |
| 2 | 3 | The paper: model in one markdown cell; run the three rules on the Section 4 cases; reproduction table; **Plot 1**. |
| 3 | 2 | Regime map from cache; **Plot 2**. |
| 4 | 2 | AI as a Cox process; regime numbers for AI; **Plot 3**. |
| 5 | 3 | ΔQ for PJM; Fermi ladder; quantile table; flip the efficiency toggle live; **Plot 4**. |
| 6 | 1 | Export the handoff files; one-sentence bridge to presenter 3. |

Constraints: the whole notebook runs top to bottom from a fresh kernel in under 2 minutes with caches, and every live cell takes under 5 seconds. Markdown cells are short and in plain English, each with the one equation that matters. Every plot title states its finding.

**Figure style:** readable on a projector (font ≥ 14 pt, lines ≥ 2 pt), colour-blind-safe palette, units on every axis, one accent colour for the point of the plot. Save each figure as PNG (200 dpi) and SVG to `outputs/figures/`.

---

## 8. Repository layout

```
ai-capacity-demo/
  PROJECT_PROMPT.md          # this file
  README.md                  # what this is, how to run, where outputs go
  WORKLOG.md                 # append-only log while working
  requirements.txt           # numpy, scipy, pandas, matplotlib, jupyter, nbconvert, pytest (+ SALib if optional part done)
  refs/                      # paper PDF (not committed) + paper_notes.md
  src/aicap/
    __init__.py
    erlang_a.py              # stationary law, E[Q], cost C(b; λ), Π(b)
    capacity_rules.py        # optimal, newsvendor, known-rate rules; gaps
    cox.py                   # compound Poisson simulation, theory curves, regime numbers
    params.py                # all priors with ranges and sources
    pjm_load.py              # epistemic sampler, P0, PT, ΔQ, ladder, quantile capacity
    plotting.py              # shared style and the four figures
  scripts/
    precompute_regime.py     # builds outputs/cache/regime_grid.npz
    run_all.py               # runs everything, writes outputs/summary.json and figures
  notebooks/demo.ipynb
  tests/
  outputs/{figures,cache,handoff}/ and outputs/summary.json
  docs/GUIDE.md
  docs/REFERENCES.md
```

Use a virtual environment. Do not install anything globally.

---

## 9. Tests and verification (all must pass)

**Erlang-A**
- When γ = μ the chain is M/M/∞, so N ~ Poisson(λ/μ); compare E[(N − b)⁺] with the Poisson value.
- When b = 0, N ~ Poisson(λ/γ) and E[Q] = λ/γ.
- Flow balance: λ = μ·E[min(N, b)] + γ·E[Q] to 1e−9 relative error.
- E[Q] decreases in b; truncation error below 1e−12.
- Cost unimodal in b for the Section 4 parameters (checked on a grid).

**Rules**
- Newsvendor gives 43, 87, 350 for the three uniforms (with the paper's rounding).
- With fixed λ, the newsvendor rule gives b = λ/μ (rounded).
- Gauss–Legendre integration converges (results stable to 1e−6 when the node count doubles).

**Cox process**
- Simulated mean and variance of S match E[S] = n̄·m₁ and Var S = n̄·m₂ + n̄²·m₁²·CV_rate² within Monte Carlo error (z-test at the 0.1% level).
- Lognormal moment formulas.

**PJM model**
- With every parameter set to its median, the Fermi ladder ends near the prototype value (section 11, tolerance 2%).
- The quantiles are stable when the seed changes (within the bootstrap standard error).
- Units: a hand calculation of 2.5×10⁹ × 0.34 Wh ÷ 24 h = 35.4 MW is reproduced.

**End to end**
- `python scripts/run_all.py` completes.
- `jupyter nbconvert --to notebook --execute notebooks/demo.ipynb` completes without errors.
- Open each saved figure and check that nothing is clipped, every axis has units, and the title states the finding.

---

## 10. Documentation (the precise guide I need)

### `docs/GUIDE.md`

Written for an applied-math student who will present this and answer questions on it. Precise, not padded. Every number comes from `outputs/summary.json`; put the run timestamp and git commit hash at the top.

1. **Overview (half a page).** The question, the answer in two sentences, and a mermaid diagram of the pipeline: presenter 1 (Wh per query) → Cox model and regime test → PJM ΔQ distribution → quantile capacity → presenter 3 (price).
2. **Knowledge map.** A table with one row per concept:
   `# | Concept | Precise statement (equation) | Where used (file::function and notebook cell) | Why it is needed | Source (author year, section or page) | How it was verified (test name or check)`.
   Then one subsection per concept with a 5–15 line derivation or explanation. Cover at least:
   1. Poisson process; superposition and thinning
   2. Doubly stochastic (Cox, mixed) Poisson process; overdispersion Var N = E[Λ] + Var Λ
   3. Compound Poisson sums; Wald's identity; conditional variance
   4. Laws of total expectation and total variance; aleatory vs epistemic uncertainty
   5. Lognormal distribution: moments, mean vs median, tail share 1 − Φ(z − σ)
   6. Law of large numbers and CLT scaling (CV ∝ 1/√n)
   7. Birth–death chains, detailed balance, the Erlang-A (M/M/b+M) model, flow balance
   8. Integer optimization of a unimodal cost
   9. Fluid approximation and the newsvendor problem; derivation of the critical fractile
   10. Square-root safety staffing (Halfin–Whitt / QED regime) and when it applies
   11. Bassamboo et al. Theorem 1: the CV vs 1/√E regimes, stated exactly as in the paper
   12. Monte Carlo estimation: quantile estimation, bootstrap standard errors, seeds
   13. Rao–Blackwellization (conditional Monte Carlo)
   14. Gompertz growth vs geometric Brownian motion; the volume-vs-efficiency race
   15. Fenton–Wilkinson lognormal approximation
   16. Jensen's inequality and why the handoff is a distribution
   17. Electricity units and power concepts: Wh/day to average W, peak vs average, coincidence factor, PUE
   18. PJM capacity market basics: delivery year, forecast-based procurement, firm vs non-firm large loads
   19. The newsvendor reading of grid capacity (labelled as an analogy)
   20. Sobol indices (only if the optional part was done)
3. **What was done.** A chronological table distilled from `WORKLOG.md`:
   `Step | Goal | What was done | Inputs | Output files | Key result (numbers) | Checks passed | Decisions and why`.
4. **Results.** Reproduction table vs the paper; regime-map findings; AI's regime numbers; ΔQ quantile tables for both delivery years and both efficiency structures; the Fermi ladder; cost of uncertainty; quantile capacity for each cost ratio; implied quantile of PJM's ~30 GW.
5. **Assumptions and limitations.** For each: what it is, where it enters, how much it matters (with a number where possible), and what would change it. Include: the paper's stationary single-rate model vs multi-year growth; PJM's demand curve vs a pure newsvendor; independence of priors; judgment-based priors; AI-only ΔQ vs PJM's all-data-centre number.
6. **Discrepancies and open items.** The reproduction mismatch (if any) and how it was investigated; every remaining VERIFY item.
7. **How to reproduce.** Exact commands.
8. **Talk script.** For each of the four plots: what to say (at most three sentences) and the one takeaway. Then the five most likely audience questions with short answers.
9. **Glossary.**

### `docs/REFERENCES.md`

Full citation, link, what it is used for in this project, and status (`checked` or `VERIFY`) for every source in section 12. Do not add sources you have not opened.

---

## 11. Reference values from a prototype (for sanity checks only)

These came from a quick prototype with the same priors (400,000 draws, seed 1). Your numbers should be close; if they differ by more than ~10%, explain why in `WORKLOG.md`.

**ΔQ (GW)**

| Delivery year | Efficiency | P10 | P50 | P90 | Mean |
| --- | --- | --- | --- | --- | --- |
| 2028/29 (T = 3) | matures (default) | 0.82 | 4.17 | 22.2 | 10.5 |
| 2028/29 (T = 3) | compounding | 0.18 | 1.99 | 13.7 | 6.39 |
| 2030/31 (T = 5) | matures (default) | 0.94 | 5.42 | 34.2 | 16.95 |
| 2030/31 (T = 5) | compounding | −0.19 | 0.97 | 13.07 | 6.89 |

- P(ΔQ > 30 GW), DY 2030/31 default: 0.116. Baseline 2025 AI peak load in PJM, P₀ median ≈ 0.49 GW.
- Fermi ladder, medians, DY 2030/31 default (MW): 35 → 56 (mean, overhead) → 449 (all providers) → 1,187 (2025 mix) → 10,523 (volume growth) → 32,283 (mix shift) → 12,854 (efficiency) → 20,718 (training) → 4,140 (PJM share) → 4,763 (summer peak).
- Log-scale Sobol first-order indices, 2030 default: growth 0.31, mix 0.31, base volume 0.17, efficiency 0.09, Wh per query 0.05, training 0.03, siting and peak 0.02.
- Cox crossover n̄* = e^(σ²)/CV_rate² with CV_rate = 1: about 3 queries (σ = 1), 55 (σ = 2), 8,103 (σ = 3).

---

## 12. Sources

**Checked (opened during preparation):**
- Bassamboo, Randhawa, Zeevi (2010), *Management Science* 56(10):1668–1686, doi:10.1287/mnsc.1100.1203 — primary reference.
- Oviedo et al. (2026), "Energy use of AI inference, efficiency pathways, and test-time scaling," *Joule* 102430, doi:10.1016/j.joule.2026.102430, arXiv:2509.20241. Median 0.31 Wh (IQR 0.16–0.60); test-time scaling 3.91 Wh; β = 1.31 (Supplemental Note S5); 0.73 GWh per 10⁹ queries/day; 8–20× line-of-sight efficiency.
- Elsworth et al. (2025), "Measuring the environmental impact of delivering AI at Google scale," arXiv:2508.15734. Median Gemini Apps text prompt 0.24 Wh including idle capacity; 33× reduction in one year.
- Bai et al. (2026), "How do AI agents spend your money? Analyzing and predicting token consumption in agentic coding tasks," arXiv:2604.22750. Agentic ~1000× tokens vs code chat and reasoning, input-dominated; runs on the same task differ up to 30×.
- Luccioni & Gamazaychikov (2025), "AI Energy Score v2," Hugging Face blog, https://huggingface.co/blog/sasha/ai-energy-score-v2. Reasoning models ~30× energy on average; 150–700× for the same model with reasoning on; GPU energy.
- PJM (2026), 2026 Load Forecast Report, https://www.pjm.com/-/media/DotCom/library/reports-notices/load-forecast/2026-load-report.pdf. Summer peak 222,106 MW in 2036, +65,733 MW over 10 years; near-term firm vs non-firm large loads.
- PJM Load Analysis Subcommittee (24 Nov 2025), Load Adjustment Requests Summary, https://www.pjm.com/-/media/DotCom/committees-groups/subcommittees/las/2025/20251124/20251124-item-03---large-load-adjustment-requests-summary.pdf. PJM ≈ 40% of US data centres in 2025; supports up to ~30 GW PJM data-centre growth 2025–2030; 70% utilization; minimum 36-month ramps; non-firm zero before 2030 and halved from 2030 in most zones.
- IEA (2025), *Energy and AI*, https://www.iea.org/reports/energy-and-ai/executive-summary. US 45% of global data-centre electricity in 2024.
- Google I/O 2026 keynote blog (19 May 2026): 9.7 T tokens/month (May 2024), ~480 T (May 2025), >3.2 quadrillion (May 2026).
- TechCrunch (21 Jul 2025): ChatGPT 2.5 B prompts/day.
- Norris et al. (2025), *Rethinking Load Growth*, Nicholas Institute, Duke University. PJM 18 GW of curtailment-enabled headroom at 0.5% annual curtailment.
- Eick, Massey, Whitt (1993), "The physics of the M_t/G/∞ queue," *Operations Research* 41(4):731–742.
- Jongbloed & Koole (2001), "Managing uncertainty in call centres using Poisson mixtures," *Appl. Stoch. Models Bus. Ind.* 17(4):307–318, doi:10.1002/asmb.444.
- Hoffman & Hammonds (1994), *Risk Analysis* 14(5):707–712, doi:10.1111/j.1539-6924.1994.tb00281.x (aleatory vs epistemic, two-level Monte Carlo).
- Der Kiureghian & Ditlevsen (2009), "Aleatory or epistemic? Does it matter?" *Structural Safety* 31(2):105–112.
- Fenton (1960), "The sum of log-normal probability distributions in scatter transmission systems," *IRE Trans. Commun. Syst.* 8(1):57–67.
- Sobol' (2001), *Math. Comput. Simul.* 55:271–280; Saltelli et al. (2010), *Comput. Phys. Commun.* 181(2):259–270 (optional part only).

**VERIFY before citing:**
- Every number in the Section 4 table above (extracted by an automated reader).
- The exact statement of Theorem 1 and the cost convention in §3.
- Patterson et al. (2022), *Computer* 55(7):18–28: the ~3/5 inference vs 2/5 training split.
- S. M. Ross, *Introduction to Probability Models* (12th ed.): section numbers for compound and mixed Poisson processes.
- Halfin & Whitt (1981), "Heavy-traffic limits for queues with many exponential servers," *Operations Research* 29(3): pages.

---

## 13. Definition of done

- [ ] `refs/paper_notes.md` written from the actual PDF, with page numbers.
- [ ] All tests pass; `scripts/run_all.py` and the notebook run cleanly from scratch.
- [ ] Four figures saved, each checked visually.
- [ ] Reproduction table vs the paper, with any mismatch explained.
- [ ] Handoff CSVs, summary JSON and README in `outputs/handoff/`.
- [ ] `docs/GUIDE.md` complete per section 10, every number traceable to `outputs/summary.json`.
- [ ] `docs/REFERENCES.md` complete; no unverified source presented as checked.
- [ ] `WORKLOG.md` complete; git history with milestone commits.
- [ ] Final chat report: what was built, key numbers, deviations, open items.
