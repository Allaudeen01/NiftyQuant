# EXP034-A — Phase 3 Defect Disclosure

Both defects below were found **during Phase 3 execution and corrected before
any result was reported**. Neither altered the locked methodology: the binding
preregistration (`18a178e`) and its config remained byte-identical throughout,
verified by `git diff` against the lock commit at every step.

Recorded per `docs/RESEARCH_POLICY.md` §1 — defects are documented, never
quietly patched.

---

## Defect 1 — HAC one-sided test returned NaN on a valid series

**Affected component.** `nifty_quant/analytics/clark_west.py::hac_one_sided`,
the Newey-West cross-check on the Clark-West statistic. (The *primary*
inference is the day-block bootstrap, which was unaffected.)

**Mechanism.** The degeneracy guard was written as
`np.allclose(np.std(x), 0.0)`. `np.allclose` applies a default **absolute**
tolerance of `1e-8`. A genuine Clark-West series on this data has a standard
deviation of order `1e-10` — three orders of magnitude *below* that tolerance —
so a perfectly well-conditioned series was declared constant and the function
returned `NaN` instead of a test statistic.

**Symptom observed.** `HAC p = nan` for three of the twelve comparisons in the
first execution, including the T2a and T2b primary comparisons.

**Correction.** Degeneracy is now tested exactly (`np.std(x) == 0.0`). The
comment records why the tolerant form is wrong here, so it is not reintroduced.

**Effect on results.** None on any primary verdict — the primary inference was
always the bootstrap. After correction the HAC cross-check agrees with the
bootstrap on every comparison (e.g. T2a MODEL_3 vs MODEL_2: bootstrap 0.1174,
HAC 0.0891 — both non-significant).

**Methodology unchanged.** The locked config already designated the day-block
bootstrap as primary and HAC as a cross-check; that assignment was not touched.

---

## Defect 2 — verdict logic omitted the locked MDE clause

**Affected component.** `scripts/exp034_walk_forward.py`, the code assigning
the locked verdict labels.

**Mechanism.** The locked inconclusive band requires **two** conditions
jointly: *"point estimate positive but the confidence interval spans zero **and**
the estimate is below the pre-computed minimum detectable effect."* The first
implementation tested only the confidence-interval condition and ignored the
MDE condition entirely. Every target whose CI spanned zero therefore collapsed
to `INCONCLUSIVE - UNDERPOWERED`, regardless of whether the study actually had
power to detect the effect in question.

**Symptom observed.** All four primary targets were labelled
`INCONCLUSIVE - UNDERPOWERED` in the first execution.

**Correction.** The Clark-West mean is now expressed as an approximate
incremental R-squared (`cw_mean / var(y_oos)`) so it is dimensionally
comparable to the pre-computed `mde_delta_r2_approx`, and both clauses are
evaluated. The power addendum now also emits a machine-readable
`power_addendum_values.csv`; the walk-forward **aborts** if it is absent rather
than silently skipping the clause.

**Effect on results — material, and in the direction of a stronger claim.**

| target | first execution | corrected | why |
|---|---|---|---|
| T1a | INCONCLUSIVE — UNDERPOWERED | unchanged | dR2 0.00019 < MDE 0.00119 |
| T1b | INCONCLUSIVE — UNDERPOWERED | unchanged | dR2 0.00143 < MDE 0.00264 |
| **T2a** | INCONCLUSIVE — UNDERPOWERED | **NO EXPLORATORY EVIDENCE** | dR2 0.00214 **>** MDE 0.00170 |
| T2b | INCONCLUSIVE — UNDERPOWERED | unchanged | dR2 0.00050 < MDE 0.00258 |

T2a's point estimate exceeds its minimum detectable effect, so the study had
adequate power there and still found nothing significant. The corrected label
is the stronger and more honest statement; the defect had caused it to be
*understated*, not overstated.

**Methodology unchanged.** The MDE values were computed from the first 30
training days only and recorded **before any fold was scored**, exactly as the
lock requires. The correction applied the locked decision rule as written; it
did not modify, relax or reinterpret that rule.

---

## Verification

At every step after correction:

- `git diff 18a178e -- docs/preregistrations/exp034_option_information_content.md docs/preregistrations/exp034_config.json` → empty (locked files byte-identical)
- `python scripts/exp034_preflight.py --panel ... --features ...` → PASS
- `python scripts/exp033_freeze.py --verify` → VERIFIED

The corrected execution is the one of record and is fingerprinted in
`docs/exp034a_FREEZE.json`.
