# Research Policy — Standing Rules

Platform-level rules that apply across experiments. Individual experiments may
be *stricter* than this document; they may never be looser.

---

## 1. Post-freeze defects

> **A defect discovered after an experiment is frozen may be documented, and
> may be used to motivate future experiments, but may not be repaired inside
> the frozen experiment, and may not be used to retroactively change its
> verdict.**

A frozen experiment is a faithful record of *the exact hypothesis, the exact
implementation, the exact data, and the exact result*. That record retains its
value even when the implementation is later found to be flawed — indeed, the
record is what makes the flaw discoverable at all. Repairing it in place would
destroy the only evidence of what was actually run.

When a post-freeze defect is found:
1. Leave every frozen artifact untouched — code, reports, preregistration,
   commit, freeze manifest, tag.
2. Record the defect in the *successor* experiment's preregistration, or in a
   non-binding errata document, stating precisely what was implemented versus
   what was documented.
3. Let the correction motivate the successor's design.
4. Do not restate the frozen experiment's verdict. Its verdict stands as the
   result of what was actually run.

**Applied instances**
- **EXP033 / E1 volume family.** The implementation applied a trailing z-score
  to the *raw cumulative daily volume level* while its docstring described a
  "2-minute increment". This weakens the substantive interpretation of E1 but
  does not justify unfreezing. EXP034 uses first differences instead and
  records the distinction. EXP033's verdict is unchanged.

---

## 2. Preregistration amendment

A locked preregistration may not be edited. Factual errata are recorded in a
separate non-binding errata document; methodological changes require a new
experiment id.

Sensitivity analyses declared *before the first out-of-sample fold is scored*
are permitted as additions to reporting, provided they are constrained so they
can only **invalidate** a positive finding, never create or rescue one. Any
analysis declared after results are seen belongs to a new experiment.

---

## 3. Data-generating process versioning

Data collected under materially different collection semantics may not be
silently pooled. When a collector or provider change alters **information
timing, field meaning, or feature observability**, the affected observations
constitute a new `dgp-v<N>` and require either a compatibility ruling made
before results are inspected, or a new experiment.

Changes to collection code must be versioned and logged with their effective
date in `docs/data_versions.md`.

**Corollary, accepted explicitly:** fixing a genuine collection defect may
invalidate pooling with previously collected data and so delay a pending
confirmatory replication. The science should not preserve a bad
data-generating process merely to reach a day count sooner — but the trade-off
must be made deliberately, not by accident.

---

## 4. Evidence vocabulary

"No evidence of an effect" and "evidence of no effect" are different
conclusions and must never be conflated. A null result may be reported as
evidence of *absence* only when the pre-computed minimum detectable effect is
smaller than the smallest effect that would be meaningful. Otherwise the
correct report is "no evidence at this sample size", accompanied by the power
analysis.

---

## 5. Sample reuse across experiments

Repeated analysis of the same sample across successive experiments is a
program-level snooping exposure that no within-experiment correction removes.
An experiment that is the Nth pass over an existing sample is exploratory by
construction, and a confirmatory claim requires either a sealed holdout of
adequate power or replication on forward-collected data.
