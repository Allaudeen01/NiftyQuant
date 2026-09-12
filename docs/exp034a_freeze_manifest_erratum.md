# EXP034-A — Freeze-Manifest Erratum

**Subject:** `docs/RESEARCH_POLICY.md` was incorrectly included as a tracked
experiment source in EXP034-A's freeze manifest.

**Classification:** *freeze-manifest design defect* — **not** a change to any
EXP034-A experimental artifact.

**Resolution:** documented here. The freeze manifest is **not** rewritten, the
tag is **not** moved, and the standing policy rule is **not** reverted.

---

## 1. What went wrong

`scripts/exp034a_freeze.py` lists `docs/RESEARCH_POLICY.md` in
`TRACKED_SOURCES`, so its content hash was recorded in
`docs/exp034a_FREEZE.json` at freeze time and is re-checked on every
`--verify`.

## 2. Why that was the wrong choice

`RESEARCH_POLICY.md` is a **living platform-level governance document**. Its
entire purpose is to accumulate standing rules as the programme learns — §1
(post-freeze defects) came from EXP033, §2 (contract-identity differencing)
came from EXP035, and more will follow.

A document designed to change is structurally unsuitable for immutable
experiment-source hashing. Pinning it inside one experiment's manifest
guarantees that *every future policy addition* breaks that experiment's
verification, producing a false alarm that says "this experiment is no longer
reproducible" when nothing about the experiment has changed.

The correct design is for an experiment manifest to fingerprint only artifacts
that define *that experiment*: its preregistration, config, code, data and
outputs. Cross-cutting governance documents should be versioned independently.
EXP033's manifest does not track the policy file; only EXP034-A's does.

## 3. Sole cause of the current drift

The post-freeze addition of **§2 Contract-Identity Differencing** to
`RESEARCH_POLICY.md`. That rule was added deliberately and on instruction,
after EXP035's Phase 0 audit showed that differencing *after* relative-strike
normalization silently subtracts two different contracts whenever the ATM
strike rolls (10.71% of snapshots).

```
DRIFT  docs/RESEARCH_POLICY.md
  expected: 6d684544adcdb2ccdcc13146ee0bf2c41d68c491a885e6b1c1355da3eaa0ddf9
  actual:   dc4a6651ce1654678456c7a019ffd23a7d40777fa5ead3127958291dad5b322e
```

This is the **only** drifting item. No other tracked source, artifact or output
differs.

## 4. What remains independently verified

Every one of the following re-hashes **byte-identically** against
`docs/exp034a_FREEZE.json`:

| Artifact class | Status |
|---|---|
| EXP034-A preregistration (`exp034_option_information_content.md`) | VERIFIED |
| Locked config (`exp034_config.json`) | VERIFIED |
| Feature registry (15 features, formulas, ablation groups) | VERIFIED |
| Fold definitions (walk-forward structure) | VERIFIED |
| Experimental source files (pipeline, features, Clark-West, preflight) | VERIFIED |
| Source data fingerprint (59 days, 9,816 files) + India VIX | VERIFIED |
| Derived panels (observations, feature panel, certification) | VERIFIED |
| Out-of-sample predictions | VERIFIED |
| Statistical outputs (primary, secondary, fold, per-day) | VERIFIED |
| Reports (phase 1/2/3, power addendum) | VERIFIED |

The scientific record of EXP034-A is intact. Its verdicts stand unchanged:
no incremental option-chain information beyond MODEL_2; one adequately powered
primary negative (T2a); three underpowered/inconclusive; strong secondary
evidence that India VIX improves short-horizon realized-movement prediction.

## 5. The freeze must not be rewritten

`docs/exp034a_FREEZE.json` is **not** regenerated, and tag
`EXP034_A_FINAL_FROZEN` is **not** moved. Rewriting a manifest to make a
verification pass would destroy the very property the manifest exists to
provide. Per `RESEARCH_POLICY.md` §1, a post-freeze defect is documented and
may motivate successors — it is never repaired in place.

The standing §2 rule is likewise **not** reverted or relocated merely to
restore an old hash. The rule is correct and load-bearing; the manifest's
inclusion of the file was the error.

## 6. How verification must now be read

**The warning is not suppressed.** `python scripts/exp034a_freeze.py --verify`
continues to exit non-zero and to report the drift, exactly as designed. That
tool is itself a tracked source and is therefore left unmodified.

A companion tool, `scripts/exp034a_verify_artifacts.py`, reports the same
checks **categorized**:

```
EXPERIMENTAL ARTIFACTS: VERIFIED
POLICY FILE: EXPECTED DOCUMENTED DRIFT
  CAUSE: historical freeze-manifest inclusion of living RESEARCH_POLICY.md
```

It exits non-zero if **any experimental artifact** drifts, and exits zero only
when the sole drift is the documented policy file. It never hides a real
failure — it distinguishes two kinds of failure that must not be confused.

## 7. Guidance for future manifests

Do not include living governance documents in an experiment's
`TRACKED_SOURCES`. Fingerprint only what defines the experiment. If a policy
document's state matters to an experiment, record the **policy version or
commit** rather than a content hash of a file expected to change.

---
*Documented, not repaired. EXP034-A remains frozen and immutable.*
