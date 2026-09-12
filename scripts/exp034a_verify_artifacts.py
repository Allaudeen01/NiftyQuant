"""EXP034-A categorized verification — separates artifact drift from the
documented policy-file manifest defect.

Background: `docs/RESEARCH_POLICY.md` was incorrectly included in EXP034-A's
freeze manifest as a tracked experiment source. It is a LIVING platform-level
governance document, so every future standing-rule addition breaks that
manifest's verification even though no experimental artifact changed. See
`docs/exp034a_freeze_manifest_erratum.md`.

This tool exists because two different failures must not be confused:

  * an EXPERIMENTAL ARTIFACT drifting  -> the experiment is not reproducible
  * the POLICY FILE drifting           -> the documented manifest design defect

It does NOT suppress anything. `scripts/exp034a_freeze.py --verify` is left
completely unmodified (it is itself a tracked source) and continues to exit
non-zero and report the drift. This companion only categorizes.

Exit codes:
    0  experimental artifacts verified; the only drift is the documented policy file
    1  an experimental artifact drifted -> genuine reproducibility failure
    2  manifest missing

    python scripts/exp034a_verify_artifacts.py

Read-only. Never writes, never regenerates the manifest, never moves a tag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp034a_freeze as F  # noqa: E402  (read-only import; file unmodified)

POLICY_FILE = "docs/RESEARCH_POLICY.md"
ERRATUM = "docs/exp034a_freeze_manifest_erratum.md"


def main() -> int:
    if not F.MANIFEST.exists():
        print(f"No manifest at {F.MANIFEST}.")
        return 2
    old = json.loads(F.MANIFEST.read_text(encoding="utf-8"))

    artifact_drift: list[str] = []
    policy_drift: list[str] = []

    print("=" * 84)
    print("EXP034-A CATEGORIZED VERIFICATION")
    print(f"manifest    : {F.MANIFEST}")
    print(f"frozen at   : {old.get('frozen_at_utc')}")
    print(f"tag         : {old.get('tag')}")
    print("=" * 84)

    def cmp(label: str, expect, actual, bucket: list):
        if expect != actual:
            bucket.append(label)
            print(f"  DRIFT  {label}")
        else:
            print(f"  OK     {label}")

    print("\n--- binding preregistration and config ---")
    cmp("preregistration", old["binding_preregistration"]["sha256"],
        F.sha256_source(Path(F.PREREG_MD)), artifact_drift)
    cmp("locked config", old["binding_config"]["sha256"],
        F.sha256_source(Path(F.CONFIG_JSON)), artifact_drift)

    print("\n--- feature registry and fold definitions ---")
    cmp("feature registry", old["feature_registry"], F.feature_registry(), artifact_drift)
    cmp("fold definitions", old["fold_definitions"], F.fold_definitions(), artifact_drift)

    print("\n--- tracked source files ---")
    cur = F.source_files()
    for rel, h in old["source_files_sha256"].items():
        if rel.startswith("_"):
            continue
        bucket = policy_drift if rel == POLICY_FILE else artifact_drift
        cmp(rel, h, cur.get(rel), bucket)

    print("\n--- source data (option chains + India VIX) ---")
    sd = F.source_data_manifest()
    for k in ("n_days", "n_files", "root_sha256", "vix_n_files", "vix_root_sha256"):
        cmp(f"source_data.{k}", old["source_data"][k], sd[k], artifact_drift)

    print("\n--- derived panels ---")
    cur_d = F._hash_list(F.DERIVED, normalise_text=False)
    for rel, meta in old["derived_artifacts"].items():
        act = cur_d.get(rel, {"present": False})
        if not act.get("present"):
            print(f"  WARN   {rel} absent (rebuildable)")
            continue
        for k in ("n_rows", "n_columns"):
            cmp(f"{rel} {k}", meta.get(k), act.get(k), artifact_drift)

    print("\n--- predictions, statistics, reports ---")
    cur_o = F._hash_list(F.OUTPUTS, normalise_text=True)
    for rel, meta in old["outputs"].items():
        act = cur_o.get(rel, {"present": False})
        if not act.get("present"):
            print(f"  WARN   {rel} absent (regenerate to re-verify)")
            continue
        cmp(rel, meta.get("sha256"), act.get("sha256"), artifact_drift)

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 84)
    if artifact_drift:
        print("EXPERIMENTAL ARTIFACTS: **DRIFT DETECTED**")
        for a in artifact_drift:
            print(f"    - {a}")
        print("\nEXP034-A is NOT reproducible from this tree. This is a genuine")
        print("reproducibility failure, NOT the documented policy-file defect.")
        print("Do NOT edit EXP034-A. Publish the change as a new experiment.")
        print("=" * 84)
        return 1

    print("EXPERIMENTAL ARTIFACTS: VERIFIED")
    print("    preregistration, config, feature registry, fold definitions,")
    print("    experimental sources, source data, derived panels, predictions,")
    print("    statistics and reports all match the freeze byte-for-byte.")

    if policy_drift:
        print("\nPOLICY FILE:")
        print("    EXPECTED DOCUMENTED DRIFT")
        print("    CAUSE: historical freeze-manifest inclusion of living "
              f"{POLICY_FILE}")
        print(f"    See {ERRATUM}")
        print("\n    This drift is EXPECTED and is NOT suppressed: "
              "`exp034a_freeze.py --verify`")
        print("    still exits non-zero and reports it. The manifest is not "
              "rewritten and")
        print("    the tag is not moved.")
    else:
        print("\nPOLICY FILE: matches the frozen hash (no drift).")

    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
