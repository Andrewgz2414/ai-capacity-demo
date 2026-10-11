# References

Every source listed in spec §12, nothing added. Status key:

- **checked (this build)**: opened and the cited content confirmed while building this project (what was confirmed is stated).
- **checked (author)**: listed as checked in spec §12 (opened by the author during preparation); not re-opened in this build. Numbers quoted from it in this project are as given in the spec.
- **VERIFY**: not confirmed; do not cite the detail in question until checked.

## Primary reference

| Source | Link | Used for | Status |
|---|---|---|---|
| A. Bassamboo, R. S. Randhawa, A. Zeevi (2010). "Capacity sizing under parameter uncertainty: safety staffing principles revisited." *Management Science* 56(10):1668–1686. doi:10.1287/mnsc.1100.1203 | [PDF (working paper, 10 May 2010)](https://business.columbia.edu/sites/default/files-efs/pubfiles/5051/Bassamboo-Randhawa-Zeevi-parameter-uncertainty.pdf) · [RePEc record](https://ideas.repec.org/a/inm/ormnsc/v56y2010i10p1668-1686.html) | Model, cost (eq. 1), newsvendor prescription (eqs. 2–3), §4 numerical setup and tables, Theorem 1, Propositions 1–3 | **checked (this build)**: all 37 pages read; notes with page numbers in `refs/paper_notes.md`. The spec's §4 table values match the PDF's Tables 2–3 exactly; Theorem 1 quoted verbatim (p.15); cost convention confirmed (p.7). Page numbers refer to the working-paper PDF, not the journal version. |

## Energy per query, AI volumes and mix

| Source | Link | Used for | Status |
|---|---|---|---|
| C. Elsworth et al. (2025). "Measuring the environmental impact of delivering AI at Google scale." arXiv:2508.15734 | [arXiv:2508.15734](https://arxiv.org/abs/2508.15734) | Prior for M_c (Google's 0.24 Wh median text prompt); the 33× one-year reduction (efficiency context); PUE/idle boundary of the 0.24 Wh figure (β discussion) | **checked (this build)**: abstract of the local PDF (v1) confirms the 0.24 Wh median Gemini Apps text prompt, a methodology including "AI accelerator power, host system energy, idle machine capacity, and data center energy overhead", and the 33x reduction over one year |
| Oviedo et al. (2026). "Energy use of AI inference, efficiency pathways, and test-time scaling." *Joule* 102430. doi:10.1016/j.joule.2026.102430; arXiv:2509.20241 | [arXiv:2509.20241](https://arxiv.org/abs/2509.20241) | Median 0.31 Wh (IQR 0.16–0.60) → M_c and σ_c priors; β = 1.31 (Supplemental Note S5); reasoning multiplier r (13× energy for 15× tokens); 8–20× efficiency (h prior); κ = 1.235 (S5) | checked (author) |
| Bai et al. (2026). "How do AI agents spend your money? Analyzing and predicting token consumption in agentic coding tasks." arXiv:2604.22750 | [arXiv:2604.22750](https://arxiv.org/abs/2604.22750) | Agentic multiplier a (~1000× tokens, input-dominated; runs on the same task differ up to 30×) | checked (author) |
| S. Luccioni, B. Gamazaychikov (2025). "AI Energy Score v2." Hugging Face blog | [huggingface.co/blog/sasha/ai-energy-score-v2](https://huggingface.co/blog/sasha/ai-energy-score-v2) | Reasoning ~30× energy on average (upper end of r); most newer models not more efficient (h prior) | checked (author) |
| TechCrunch (21 Jul 2025): ChatGPT 2.5 B prompts/day | — (no link in spec) | N₀ prior; Fermi ladder start; concurrency example | checked (author) |
| Google I/O 2026 keynote blog (19 May 2026): 9.7 T tokens/month (May 2024), ~480 T (May 2025), > 3.2 quadrillion (May 2026) | — (no link in spec) | N₀ prior (at an assumed 1–3 k tokens/request); g₁ prior (6.7× in a year); growth-decay λ (two-point fit) | checked (author) |
| D. Patterson et al. (2022). "The carbon footprint of machine learning training will plateau, then shrink." *Computer* 55(7):18–28 | [arXiv:2204.05149](https://arxiv.org/abs/2204.05149) | Training-to-inference ratio τ | **checked (this build)**: arXiv version, §7 and Conclusion: "Across all three years, about ⅗ of ML energy use is for inference and ⅖ for training" (Google, one week of April 2019, 2020, 2021; includes data-centre overheads), i.e. τ ≈ 0.67. Volume 55, issue 7 (July 2022) as shown for the IEEE Xplore listing in web-search results (listing not opened); page range 18–28 as given in the spec, not re-checked |

## Grid, PJM and data-centre load

| Source | Link | Used for | Status |
|---|---|---|---|
| PJM (2026). *2026 Load Forecast Report* | [pjm.com … 2026-load-report.pdf](https://www.pjm.com/-/media/DotCom/library/reports-notices/load-forecast/2026-load-report.pdf) | Summer peak 222,106 MW in 2036, +65,733 MW over 10 years → 2026 forecast peak ≈ 156 GW (physical yardstick in Fig. 4); firm vs non-firm large loads | checked (author) |
| PJM Load Analysis Subcommittee (24 Nov 2025). *Load Adjustment Requests Summary* | [pjm.com … 20251124-item-03 … .pdf](https://www.pjm.com/-/media/DotCom/committees-groups/subcommittees/las/2025/20251124/20251124-item-03---large-load-adjustment-requests-summary.pdf) | s_PJM prior (PJM ≈ 40% of US data centres); the ~30 GW data-centre growth yardstick 2025–2030; 70% utilization (κ ≈ 1.0); 36-month ramps; non-firm treatment | checked (author) |
| IEA (2025). *Energy and AI* | [iea.org/reports/energy-and-ai/executive-summary](https://www.iea.org/reports/energy-and-ai/executive-summary) | s_US prior (US 45% of 2024 data-centre electricity) | checked (author) |
| Norris et al. (2025). *Rethinking Load Growth*. Nicholas Institute, Duke University | — (no link in spec) | Context for flexible load: 18 GW of curtailment-enabled headroom in PJM at 0.5% annual curtailment (not used in a calculation) | checked (author) |

## Methods

| Source | Link | Used for | Status |
|---|---|---|---|
| S. Halfin, W. Whitt (1981). "Heavy-traffic limits for queues with many exponential servers." *Operations Research* 29(3):567–588. doi:10.1287/opre.29.3.567 | [RePEc record](https://ideas.repec.org/a/inm/oropre/v29y1981i3p567-588.html) · [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/opre.29.3.567) | Square-root safety staffing / QED regime (knowledge map #10) | **checked (this build)**: volume 29, issue 3, pages 567–588 (June 1981) from the bibliographic record; also consistent with the primary reference's bibliography (p.24). Full text not opened |
| S. G. Eick, W. A. Massey, W. Whitt (1993). "The physics of the M_t/G/∞ queue." *Operations Research* 41(4):731–742 | — | Limitations: time-varying arrival rates (§5 of the guide) | checked (author) |
| G. Jongbloed, G. Koole (2001). "Managing uncertainty in call centres using Poisson mixtures." *Appl. Stoch. Models Bus. Ind.* 17(4):307–318. doi:10.1002/asmb.444 | — | Mixed Poisson arrivals (knowledge map #2) | checked (author); bibliographic details consistent with the primary reference's bibliography (p.24) |
| F. O. Hoffman, J. S. Hammonds (1994). *Risk Analysis* 14(5):707–712. doi:10.1111/j.1539-6924.1994.tb00281.x | — | Aleatory vs epistemic uncertainty; two-level Monte Carlo (knowledge map #4) | checked (author) |
| A. Der Kiureghian, O. Ditlevsen (2009). "Aleatory or epistemic? Does it matter?" *Structural Safety* 31(2):105–112 | — | Aleatory vs epistemic uncertainty (knowledge map #4) | checked (author) |
| L. F. Fenton (1960). "The sum of log-normal probability distributions in scatter transmission systems." *IRE Trans. Commun. Syst.* 8(1):57–67 | — | Fenton–Wilkinson approximation (knowledge map #15) | checked (author) |
| I. M. Sobol' (2001). *Math. Comput. Simul.* 55:271–280 | — | Sobol indices (optional part) | checked (author) |
| A. Saltelli et al. (2010). *Comput. Phys. Commun.* 181(2):259–270 | — | Sobol estimators / sampling design used by SALib (optional part) | checked (author) |
| S. M. Ross. *Introduction to Probability Models*, 12th ed. | — | Poisson, compound Poisson and mixed Poisson processes (knowledge map #1–3) | **VERIFY**: section numbers for compound and conditional (mixed) Poisson processes not confirmed for the 12th edition (the 10th edition uses §5.4.2 and §5.4.3, per a course page found by search). Cite the chapter (ch. 5) only until checked |

## Software

numpy, scipy, pandas, matplotlib, jupyter/nbconvert, pytest, SALib (Sobol sampling and analysis), pypdf (text extraction for the notes). Exact versions in `requirements-lock.txt`.
