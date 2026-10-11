# Notes on the primary reference

A. Bassamboo, R. S. Randhawa, A. Zeevi, "Capacity Sizing Under Parameter Uncertainty: Safety Staffing Principles Revisited," *Management Science* 56(10):1668–1686 (2010), doi:10.1287/mnsc.1100.1203.

**Version read:** the working paper dated 10 May 2010, 37 pages, downloaded from
<https://business.columbia.edu/sites/default/files-efs/pubfiles/5051/Bassamboo-Randhawa-Zeevi-parameter-uncertainty.pdf>
(md5 `f2ede02434ff1af274cb197a2ce28f4c`; the PDF is in `refs/` but not committed).
**Page numbers below are this PDF's pages** (they equal the printed footers). The journal version has different pagination and may differ in detail.

Notation: the paper writes the service rate as **θ** (the spec uses μ). λ denotes the mean arrival rate EΛ.

---

## 1. Model (§3, p.7)

- b statistically identical servers; FCFS; non-idling; infinite waiting room.
- Arrivals: doubly stochastic Poisson process. The rate Λ is a random variable with distribution F and mean λ; given Λ, arrivals are homogeneous Poisson with that rate.
- Service times i.i.d. exponential with mean 1/θ, independent of the arrival process and rate.
- Each waiting customer has exponential patience τ with mean 1/γ and abandons when the waiting time in queue reaches τ. "This description corresponds to an M/M/b + M queueing model, where the arrival rate is itself random." (p.7)

## 2. Cost function (§3, eq. (1), p.7)

- Holding cost h per customer per unit time **in queue**; penalty p per abandoning customer; staffing cost c per server per unit time; planning horizon normalised to 1.
- N = number in system in steady state. Expected customer cost = (h + pγ)·E[N − b]⁺ (abandonment rate = γ·E[N − b]⁺).
- **Eq. (1):** Π(b) := (h + pγ)·E[N − b]⁺ + c·b. The expectation is over Λ as well as N (capacity is fixed before Λ is known).
- b* = minimiser, Π* = Π(b*). The paper says "we ignore integrality constraints, and assume for simplicity that b is real-valued" (p.7), yet all its tables report integer capacities.
- Footnote (p.7): E[N − b]⁺ is convex, so Π is convex with a well-defined minimiser.

**Convention check:** this is exactly the spec's convention, C(b; λ) = c·b + (h + pγ)·E_λ[Q] with Q = (N − b)⁺ and Π(b) = E_Λ[C(b; Λ)].

## 3. Newsvendor (fluid) prescription (§3, eqs. (2)–(3), pp.7–8)

- Fluid relaxation: abandonment rate ≈ E[Λ − θb]⁺; equating with γE[N − b]⁺ gives E[N − b]⁺ ≈ (1/γ)·E[Λ − θb]⁺ (p.8).
- **Eq. (2):** min_{b ≥ 0} Π̄(b) := (p + h/γ)·E[Λ − θb]⁺ + c·b. A newsvendor problem with unit capacity cost c/θ and unit "price" p + h/γ.
- **Eq. (3):** b̄ = (1/θ)·F̄⁻¹( (c/θ) / (p + h/γ) ), with F̄ = 1 − F and F̄⁻¹(y) = inf{x ∈ ℝ₊ : F̄(x) ≤ y}. So P(Λ > θb̄) = c/(θ(p + h/γ)).
- If c/θ > p + h/γ, install no capacity (p.8).
- The uncertainty hedge is proportional to the standard deviation of F (Proposition 2, p.16), unlike the square-root variability hedge.

## 4. Numerical setup (§4.1, p.9)

- "c = 1/3 and p = h = 1, … the mean service time to 1 and the mean impatience time to 1/3" ⇒ θ = 1, γ = 3.
- Fractile: (c/θ)/(p + h/γ) = (1/3)/(4/3) = **1/4**, so b̄ is the **75th percentile** of Λ (θ = 1). Queue-cost weight h + pγ = 4.
- Uniform cases U[25,50], U[50,100], U[200,400], each with CV = 1/(1.5·√12) = **19.2 %** (Table 2 caption, p.9). Deterministic cases λ = 37.5, 75, 300 = the uniform means (p.9).
- **Computation (p.10):** Π(b̄) "is assessed using the characterization of the steady-state distribution in the underlying Markov chain, conditional on the arrival rate. The expected penalty costs are then computed using numerical integration." b* "was computed numerically by evaluating the objective function for a large set of capacity levels, and utilizing the convexity of the objective function." For deterministic rates b̄ = λ/θ (p.10).
- **Rounding of b̄:** not stated. All 20 reported b̄ values (Tables 2, 3, 4, 7) equal **floor(F⁻¹(3/4)/θ)**: 43.75→43, 87.5→87, 350→350, 37.5→37, 162.5→162, 157.5→157, 152.5→152, 1.75→1, 3.5→3, 8.75→8, 17.5→17, 26.25→26. Round-to-nearest would give 44 for U[25,50], so the rule is floor.

## 5. Every number in the Section 4 tables

### Table 2 (p.9): uncertain arrival rates (CV 19.2 %)

| Λ | b* | Π* | b̄ | Π(b̄) | \|b* − b̄\| | Π(b̄) − Π* | gap |
|---|---|---|---|---|---|---|---|
| U[25,50] | 46 | 17.34 | 43 | 17.49 | 3 | 0.15 | 0.9 % |
| U[50,100] | 89 | 33.13 | 87 | 33.20 | 2 | 0.07 | 0.2 % |
| U[200,400] | 351 | 127.08 | 350 | 127.08 | 1 | 0.007 | 0.006 % |

### Table 3 (p.9): deterministic arrival rates

| λ | b* | Π* | b̄ | Π(b̄) | \|b* − b̄\| | Π(b̄) − Π* | gap |
|---|---|---|---|---|---|---|---|
| 37.5 | 42 | 15.94 | 37 | 17.67 | 5 | 1.73 | 10.9 % |
| 75 | 83 | 29.64 | 75 | 31.60 | 8 | 1.96 | 6.6 % |
| 300 | 316 | 109.01 | 300 | 112.4 | 16 | 3.39 | 3.1 % |

Text (p.10): when λ quadruples from 75 to 300 the gap roughly doubles in absolute terms; b* − b̄ ≈ 1·√λ (square-root safety capacity).

### Table 4 (p.13): impact of uncertainty magnitude (E = 150, 1/√E ≈ 8 %)

| Λ | CV (%) | b* | Π* | b̄ | Π(b̄) | \|b* − b̄\| | Π(b̄) − Π* | gap |
|---|---|---|---|---|---|---|---|---|
| U[0,300] | 57.73 | 224 | 88.34 | 225 | 88.34 | 1 | 0 | 0 % |
| U[125,175] | 9.62 | 165 | 59.06 | 162 | 59.16 | 3 | 0.10 | 0.2 % |
| U[135,165] | 5.77 | 162 | 57.40 | 157 | 57.78 | 5 | 0.38 | 0.7 % |
| U[140,160] | 3.85 | 162 | 56.78 | 155 | 57.42 | 7 | 0.64 | 1.1 % |
| U[145,155] | 1.92 | 161 | 56.40 | 152 | 57.73 | 9 | 1.33 | 2.4 % |
| Λ = 150 | 0 | 161 | 56.26 | 150 | 58.25 | 11 | 1.99 | 3.5 % |

Text (p.12): "the optimality gap is still seen to be quite small for cases where the coefficient of variation exceeds 5%."

### Table 7 (Appendix C.1, p.36): system scale (CV 19.2 %)

| Λ | b* | Π* | b̄ | Π(b̄) | \|b* − b̄\| | Π(b̄) − Π* | gap |
|---|---|---|---|---|---|---|---|
| U[1,2] | 3 | 1.43 | 1 | 5.26 | 2 | 3.83 | 267.8 % |
| U[2,4] | 5 | 2.19 | 3 | 3.33 | 2 | 1.14 | 52.1 % |
| U[5,10] | 11 | 4.32 | 8 | 5.01 | 3 | 0.71 | 16.4 % |
| U[10,20] | 20 | 7.68 | 17 | 8.01 | 3 | 0.33 | 4.3 % |
| U[15,30] | 29 | 10.95 | 26 | 11.12 | 3 | 0.17 | 1.6 % |
| U[20,40] | 37 | 14.15 | 35 | 14.26 | 2 | 0.10 | 0.7 % |
| U[25,50] | 46 | 17.34 | 43 | 17.49 | 3 | 0.15 | 0.9 % |
| U[50,100] | 89 | 33.13 | 87 | 33.20 | 2 | 0.07 | 0.2 % |
| U[200,400] | 351 | 127.08 | 350 | 127.08 | 1 | 7×10⁻³ | 0.006 % |

(Tables 5, 6 on pp.20–21 are §6.2 simulations with Erlang-2 / lognormal service and patience; Table 8 on p.37 is the single-server variant. Not reproduced here.)

## 6. Intuition (§4.2, pp.10–12)

- Case γ = θ with h + pγ normalised to 1: N ~ Poisson(λ/θ) for known λ; normal approximation gives Π(b) ≈ (λ/θ − b)⁺ + K√(λ/θ)·exp(−(λ/θ − b)²/(2λ/θ)) + c·b (eq. 4) ⇒ b* ≈ b̄ + C₁√(λ/θ), Π(b̄) − Π(b*) ∝ √(λ/θ): the square-root rule.
- Random Λ: Π(b) ≈ Π̄(b) + K·E[√(Λ/θ)·exp(−(Λ/θ − b)²/(2Λ/θ))] (eq. 5); accuracy gap Δ(b) (eq. 6) stays bounded because Λ/θ is rarely within √Λ of b.

## 7. Rule of thumb (§4.3, p.13)

"If the coefficient of variation of the arrival rate has larger magnitude than 1/√E, then the newsvendor-based prescription (3) has a very accurate performance. Square-root safety capacity corrections are anticipated to offer limited improvements here. In the other case … one may benefit from refining (3) using a square-root staffing approach." Offered load E = λ/θ Erlangs.

## 8. Theory (§5, pp.13–17)

Setting (p.13): a sequence of systems indexed by λ → ∞; F_λ, CV_λ = σ_λ/λ, offered load E_λ = λ/θ. Problem (7): Π_λ(b) := (h + pγ)E[N_λ − b]⁺ + cb (p.14). Prescription (8): b̄_λ = (1/θ)F̄_λ⁻¹((c/θ)/(p + h/γ)).

**Assumption 1 (p.14).** Λ = λ + σ_λX, where X has zero mean, unit variance and a bounded density.

**Asymptotic notation (p.14).** u_λ ≫ v_λ if u_λ/v_λ increases without bound as λ → ∞; u_λ = O(v_λ) if u_λ/v_λ is bounded; u_λ ≍ v_λ if u_λ/v_λ is bounded above and away from zero.

**Proposition 1 (p.14).** If the arrival rate is deterministic (CV_λ = 0), b̄_λ = λ/θ is O(√E_λ)-optimal: Π_λ(b̄_λ) = Π*_λ + O(√E_λ).

**Theorem 1 (Uncertainty versus stochastic variability), p.15, verbatim:**
> (a) (Uncertainty-dominated regime.) If the coefficient of variation of the arrival rate satisfies CV_λ ≫ 1/√E_λ, then the newsvendor prescription (8) is O(1/CV_λ)−optimal. That is, Π_λ(b̄_λ) = Π*_λ + O(1/CV_λ).
> (b) (Variability-dominated regime.) If the coefficient of variation of the arrival rate satisfies CV_λ ≪ 1/√E_λ, then the newsvendor prescription (8) is O(√E_λ)−optimal. That is, Π_λ(b̄_λ) = Π*_λ + O(√E_λ).
> (c) If the coefficient of variation of the arrival rate satisfies CV_λ ≍ 1/√E_λ (i.e., variability and uncertainty are balanced), then the newsvendor prescription (8) is O(√E_λ)−optimal. That is, Π_λ(b̄_λ) = Π*_λ + O(√E_λ).

Notes for the talk:
- The theorem bounds the **absolute** gap Π(b̄) − Π*. Since Π* grows like E, the relative gap is O(1/(CV·E)) in regime (a) and O(1/√E) in regime (b).
- It is an asymptotic order statement (λ → ∞), not a sharp threshold: "CV ≫ 1/√E" means the ratio CV·√E grows without bound. The constant in front of 1/√E is not specified.
- General version without Assumption 1: Theorem 3 (Appendix A, p.26), using M_λ(y) := sup over a √λ·logλ window of λf_λ(x) ∨ 1.

**Theorem 2 (O(1)-optimality), p.16.** If CV_λ is bounded away from zero, Π_λ(b̄_λ) = Π*_λ + O(1).

**Proposition 2, p.16.** If b̄_λ is bounded away from zero, b̄_λ = E_λ + O(E_λ·CV_λ). Proof (p.34) gives |b̄ − E| ≤ max{√((1−α)/α), √(α/(1−α))}·σ_λ/θ with α the critical fractile.

**Proposition 3 (robustness), p.18.** For perturbations |s_λ| = O(√E_λ), Π_λ(b̄_λ + s_λ) = Π*_λ + O(min{1/CV_λ, √E_λ}).

## 9. Robustness (§6, pp.17–21)

- §6.1: arrival-rate uncertainty flattens Π around b̄ (Figure 1, p.18).
- §6.2: general service/patience distributions; prescription from eq. (10); costs estimated by discrete-event simulation; uniform Λ integrated on a 100-point grid (p.20); results (Tables 5–6) mirror Table 4.

## 10. Reproduction outcome (our code; see `outputs/summary.json` → `partA`)

Method: exact birth–death stationary law in log space (tail mass < 1e-12), Gauss–Legendre over uniform Λ (converged to < 1e-9 when nodes double), integer search for b*. Full side-by-side tables are in `docs/GUIDE.md` §4.

- **Table 4: reproduced on all six rows** (b*, b̄ identical; Π*, Π(b̄) within 0.02). This confirms that our model, cost and integration are the paper's.
- **Tables 2, 3, 7: not reproduced exactly.** Our b* is one server lower for every uniform case, our Π* about 0.2 lower, and our Π(b̄) lower by 0.2–0.9 (and by 3.8 for U[1,2]). No single change of γ, θ, c or K matches them (see `WORKLOG.md`, 17:46–17:53), and Table 3's implied queue-cost weight differs row by row (4.86, 4.57, 4.27 vs exactly 4 for Table 4's λ = 150). They appear to come from a computation the paper does not describe.
- The qualitative result is unchanged: with uncertain rates the newsvendor gap is below 1 % (ours: 0.41 %, 0.05 %, 0.00 %); with known rates it is several percent and shrinks like 1/√E (ours: 7.3 %, 4.7 %, 2.6 %; paper: 10.9 %, 6.6 %, 3.1 %).
- Side observation: Table 8 (p.37) reports θ̄ at the 33rd percentile although its stated parameters (c = 2/3, p = h = 1, γ = 1) imply the 67th, another sign that the appendix tables were produced with settings that differ from the text.
