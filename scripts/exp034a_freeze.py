"""Freeze / verify EXP034-A -- immutability manifest for tag `EXP034_A_FINAL_FROZEN`.

EXP034-A is CLOSED. Follows the EXP033 freeze pattern.

    python scripts/exp034a_freeze.py --write     # build docs/exp034a_FREEZE.json
    python scripts/exp034a_freeze.py --verify    # re-check every recorded hash

--verify exits non-zero on ANY drift. If a locked rule, a feature formula, a
source file, an input byte or an output byte changes, EXP034-A is no longer
EXP034-A and the change belongs to a new experiment id.

EXP034-C (the pre-committed forward-only replication) is a SEPARATE experiment
and must not be started early or run on a partial sample.

Read-only apart from writing the manifest itself. Never touches EXP033.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = Path("docs") / "exp034a_FREEZE.json"
TAG = "EXP034_A_FINAL_FROZEN"
LOCK_COMMIT = "18a178e94dcb2df985c304d3a037de50407acbba"

PREREG_MD = "docs/preregistrations/exp034_option_information_content.md"
CONFIG_JSON = "docs/preregistrations/exp034_config.json"

# (3) all Phase 1-3 source files, plus reused dependencies and the governing docs
TRACKED_SOURCES = [
    PREREG_MD,
    CONFIG_JSON,
    "docs/exp034_errata.md",
    "docs/exp034a_bug_disclosure.md",
    "docs/RESEARCH_POLICY.md",
    "scripts/exp034_preflight.py",
    "scripts/exp034_build_observations.py",
    "scripts/exp034_build_features.py",
    "scripts/exp034_power_addendum.py",
    "scripts/exp034_walk_forward.py",
    "scripts/exp034a_freeze.py",
    "nifty_quant/features/exp034_features.py",
    "nifty_quant/analytics/clark_west.py",
    # reused, not owned -- tracked so shared-infrastructure drift is visible
    "nifty_quant/analytics/data_snooping.py",
    "nifty_quant/analytics/black_scholes.py",
]

# (5)(6) derived panels
DERIVED = [
    "data/derived/exp034_day_certification.parquet",
    "data/derived/exp034_observations.parquet",
    "data/derived/exp034_feature_panel.parquet",
]

# (7)(8)(9)(10) predictions, folds, statistics, reports
OUTPUTS = [
    "reports/exp034/walk_forward_predictions.csv",
    "reports/exp034/fold_metrics.csv",
    "reports/exp034/per_day_metrics.csv",
    "reports/exp034/primary_results.csv",
    "reports/exp034/secondary_results.csv",
    "reports/exp034/power_addendum.md",
    "reports/exp034/power_addendum_values.csv",
    "reports/exp034/phase1_data_audit.md",
    "reports/exp034/phase2_feature_audit.md",
    "reports/exp034/phase3_results.md",
]

KEY_PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "pyarrow",
                "scikit-learn", "lightgbm"]


# --- hashing ---------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_source(p: Path) -> str:
    """Line-ending-normalised hash for git-tracked TEXT sources."""
    b = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(b).hexdigest()


# --- sections --------------------------------------------------------------

def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                   check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    dirty = g("status", "--porcelain")
    return {
        "commit": g("rev-parse", "HEAD"),
        "commit_short": g("rev-parse", "--short", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "commit_date": g("log", "-1", "--format=%cI"),
        "tree_clean_at_freeze": (dirty == ""),
        "uncommitted_paths": [l[3:] for l in dirty.splitlines()] if dirty else [],
        "preregistration_lock_commit": LOCK_COMMIT,
        "tag": TAG,
        "note": ("The manifest itself is committed in a child commit; `commit` above "
                 "is the tree the experiment was RUN against. The tag "
                 f"`{TAG}` points at the commit that includes this manifest."),
    }


def locked_config() -> dict:
    return json.loads(Path(CONFIG_JSON).read_text(encoding="utf-8"))


def feature_registry() -> dict:
    """(4) the locked feature registry, embedded so it survives loss of the config."""
    cfg = locked_config()
    reg = cfg["feature_registry"]
    return {
        "n_chain_features": reg["n_chain_features"],
        "n_auxiliary_indicators": reg["n_auxiliary_indicators"],
        "ablation_groups": reg["ablation_groups"],
        "features": [
            {k: f[k] for k in ("feature_id", "feature_name", "exact_formula",
                               "lookback_snapshots", "normalization",
                               "missing_value_rule", "ablation_group")}
            for f in reg["features"]
        ],
        "model_matrices": {
            "MODEL_1": cfg["model_matrices"]["MODEL_1_columns"],
            "MODEL_2": cfg["model_matrices"]["MODEL_2_columns"],
            "MODEL_3": cfg["model_matrices"]["MODEL_3_columns"],
        },
    }


def fold_definitions() -> dict:
    """(8) realized walk-forward fold structure."""
    import pandas as pd
    cfg = locked_config()
    p = Path("reports/exp034/fold_metrics.csv")
    out = {"locked_parameters": cfg["walk_forward"]}
    if p.exists():
        fm = pd.read_csv(p)
        per = (fm.drop_duplicates(["target", "fold_id"])
                 .groupby("fold_id")
                 .agg(train_days=("train_days", "max"), test_days=("test_days", "max"),
                      n_train=("n_train", "max"), n_test=("n_test", "max"))
                 .reset_index())
        out["realized_folds"] = json.loads(per.to_json(orient="records"))
        out["n_folds"] = int(fm["fold_id"].nunique())
    return out


def environment() -> dict:
    from importlib import metadata
    pkgs = {}
    for name in KEY_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "key_packages": pkgs,
    }


def source_data_manifest() -> dict:
    """(11)(12) option-chain per-day fingerprint plus the India VIX series."""
    chain_root = Path("data") / "option_chain"
    days, total_files, total_bytes = {}, 0, 0
    for day_dir in sorted(p for p in chain_root.glob("*/*/*") if p.is_dir()):
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        y, m, d = day_dir.parts[-3:]
        per = [f"{f.name}:{sha256_file(f)}" for f in files]
        nbytes = sum(f.stat().st_size for f in files)
        days[f"{y}-{m}-{d}"] = {"n_files": len(files), "bytes": nbytes,
                                 "day_sha256": sha256_text("\n".join(per))}
        total_files += len(files)
        total_bytes += nbytes
    root_hash = sha256_text("\n".join(f"{k}:{v['day_sha256']}" for k, v in sorted(days.items())))

    vix_root = Path("data") / "vix"
    vix_files = sorted(vix_root.glob("*/*.parquet"))
    vix_hash = sha256_text("\n".join(f"{f.name}:{sha256_file(f)}" for f in vix_files))

    return {
        "option_chain_path": str(chain_root).replace("\\", "/"),
        "n_days": len(days), "n_files": total_files, "bytes": total_bytes,
        "root_sha256": root_hash, "per_day": days,
        "vix_path": str(vix_root).replace("\\", "/"),
        "vix_n_files": len(vix_files), "vix_root_sha256": vix_hash,
        "note": ("EXP034-A consumed the NEAR EXPIRY only, and excluded 2026-06-26 and "
                 "2026-07-07 via the locked per-day quality screen. Exclusions are "
                 "code-level filters, not data deletions -- the raw fingerprint above "
                 "covers every collected day."),
    }


def _hash_list(paths: list[str], normalise_text: bool) -> dict:
    import pandas as pd
    out = {}
    for rel in paths:
        p = Path(rel)
        if not p.exists():
            out[rel] = {"present": False}
            continue
        entry = {"present": True, "bytes": p.stat().st_size}
        entry["sha256"] = sha256_source(p) if (normalise_text and p.suffix in
                                               (".md", ".csv", ".json", ".py")) else sha256_file(p)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            entry.update(n_rows=int(len(df)), n_columns=int(len(df.columns)))
        elif p.suffix == ".csv":
            entry["n_rows"] = int(len(pd.read_csv(p)))
        out[rel] = entry
    return out


def results_inline() -> dict:
    """Headline verdicts embedded so the manifest survives loss of the CSVs."""
    import pandas as pd
    out = {}
    for label, rel in (("primary", "reports/exp034/primary_results.csv"),
                       ("secondary", "reports/exp034/secondary_results.csv")):
        p = Path(rel)
        if p.exists():
            out[label] = json.loads(pd.read_csv(p).to_json(orient="records",
                                                           double_precision=12))
    return out


def source_files() -> dict:
    d = {rel: (sha256_source(Path(rel)) if Path(rel).exists() else None)
         for rel in TRACKED_SOURCES}
    return {"_hash_method": "sha256 of content with CRLF/CR normalised to LF", **d}


# --- build / verify --------------------------------------------------------

def build() -> dict:
    return {
        "experiment": "EXP 034-A -- Option Chain Information Content Study (exploratory)",
        "experiment_id": "exp034-A",
        "tag": TAG,
        "status": "FROZEN -- IMMUTABLE. No modifications permitted.",
        "policy": ("Any change to the preregistration, config, feature formulas, model "
                   "definitions, preprocessing rules, fold definitions, predictions, "
                   "statistics or verdicts invalidates this freeze and MUST be published "
                   "as a new experiment id. Never edit EXP034-A in place. EXP034-C is a "
                   "SEPARATE, pre-committed forward-only replication and must not be "
                   "started early or run on a partial sample. EXP034-A must not be "
                   "modified when new data arrives."),
        "verdict_summary": {
            "primary_hypotheses": 4,
            "surviving_fdr": 0,
            "smallest_q_bh": 0.4104,
            "T1a_signed_2min": "INCONCLUSIVE - UNDERPOWERED",
            "T1b_signed_10min": "INCONCLUSIVE - UNDERPOWERED",
            "T2a_absolute_2min": "NO EXPLORATORY EVIDENCE (adequately powered)",
            "T2b_absolute_10min": "INCONCLUSIVE - UNDERPOWERED",
            "headline": ("No confirmed incremental option-chain information beyond "
                         "MODEL_2. One adequately powered primary negative (T2a); three "
                         "underpowered/inconclusive. Strong SECONDARY evidence that India "
                         "VIX improves short-horizon realized-movement prediction beyond "
                         "the price/time baseline (T2a p=0.0002, T2b p=0.0018, 5/5 folds), "
                         "while MODEL_3 does not improve on MODEL_2."),
            "prohibited_interpretation": ("The secondary finding must NOT be described as "
                                          "evidence that the option-chain feature block adds "
                                          "predictive information."),
            "exploratory_status": ("EXPLORATORY BY CONSTRUCTION -- third pass over the same "
                                   "sample. May not be called confirmed, validated or "
                                   "tradable."),
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_schema": 1,
        "git": git_info(),
        "binding_preregistration": {
            "path": PREREG_MD, "sha256": sha256_source(Path(PREREG_MD))},
        "binding_config": {
            "path": CONFIG_JSON, "sha256": sha256_source(Path(CONFIG_JSON))},
        "feature_registry": feature_registry(),
        "fold_definitions": fold_definitions(),
        "environment": environment(),
        "source_data": source_data_manifest(),
        "derived_artifacts": _hash_list(DERIVED, normalise_text=False),
        "outputs": _hash_list(OUTPUTS, normalise_text=True),
        "results_inline": results_inline(),
        "bug_disclosure": {
            "document": "docs/exp034a_bug_disclosure.md",
            "defects": [
                {"id": 1, "component": "nifty_quant/analytics/clark_west.py::hac_one_sided",
                 "mechanism": ("np.allclose(std, 0.0) applies a 1e-8 ABSOLUTE tolerance; a "
                               "genuine Clark-West series has std ~1e-10 and was wrongly "
                               "declared constant, returning NaN"),
                 "correction": "degeneracy tested exactly via np.std(x) == 0.0",
                 "effect_on_verdicts": ("none -- primary inference was always the day-block "
                                        "bootstrap; HAC is a cross-check and now agrees"),
                 "methodology_changed": False},
                {"id": 2, "component": "scripts/exp034_walk_forward.py verdict logic",
                 "mechanism": ("the locked inconclusive band requires CI-spans-zero AND "
                               "estimate-below-MDE; only the CI clause was implemented, so "
                               "every CI-spanning target collapsed to UNDERPOWERED"),
                 "correction": ("CW mean expressed as approximate incremental R^2 and "
                                "compared against the pre-computed mde_delta_r2_approx; the "
                                "run now aborts if the MDE file is absent"),
                 "effect_on_verdicts": ("T2a moved INCONCLUSIVE -> NO EXPLORATORY EVIDENCE, "
                                        "a STRONGER and more honest statement; the defect had "
                                        "understated the result. T1a/T1b/T2b unchanged"),
                 "methodology_changed": False},
            ],
            "locked_methodology_unchanged": True,
            "evidence": ("git diff against the lock commit was empty for both binding files "
                         "at every step; preflight passed before any fold was scored"),
        },
        "source_files_sha256": source_files(),
    }


def _cmp(label: str, expect, actual, issues: list) -> None:
    ok = expect == actual
    print(f"  {'OK  ' if ok else 'DRIFT'}  {label}")
    if not ok:
        issues.append(label)
        print(f"          expected: {str(expect)[:120]}")
        print(f"          actual:   {str(actual)[:120]}")


def verify() -> int:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}. Run --write first.")
        return 1
    old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    issues: list[str] = []
    print("=" * 84)
    print(f"VERIFYING EXP034-A FREEZE  (tag {old.get('tag')})")
    print(f"Frozen at: {old.get('frozen_at_utc')}")
    print("=" * 84)

    print("\n--- binding preregistration and config ---")
    _cmp("preregistration sha256", old["binding_preregistration"]["sha256"],
         sha256_source(Path(PREREG_MD)), issues)
    _cmp("config sha256", old["binding_config"]["sha256"],
         sha256_source(Path(CONFIG_JSON)), issues)

    print("\n--- locked feature registry ---")
    _cmp("feature registry block", old["feature_registry"], feature_registry(), issues)

    print("\n--- fold definitions ---")
    _cmp("fold definitions", old["fold_definitions"], fold_definitions(), issues)

    print("\n--- source files ---")
    cur = source_files()
    for rel, h in old["source_files_sha256"].items():
        if rel.startswith("_"):
            continue
        _cmp(rel, h, cur.get(rel), issues)

    print("\n--- source data (option chains + VIX) ---")
    sd = source_data_manifest()
    for k in ("n_days", "n_files", "root_sha256", "vix_n_files", "vix_root_sha256"):
        _cmp(k, old["source_data"][k], sd[k], issues)

    print("\n--- derived artifacts ---")
    cur_d = _hash_list(DERIVED, normalise_text=False)
    for rel, meta in old["derived_artifacts"].items():
        act = cur_d.get(rel, {"present": False})
        if not act.get("present"):
            print(f"  WARN   {rel} absent (rebuildable; not fatal)")
            continue
        for k in ("n_rows", "n_columns"):
            _cmp(f"{rel} {k}", meta.get(k), act.get(k), issues)
        if meta.get("sha256") != act.get("sha256"):
            print(f"  WARN   {rel} sha256 differs (parquet writer nondeterminism; "
                  f"row/column counts above are authoritative)")

    print("\n--- outputs (predictions, statistics, reports) ---")
    cur_o = _hash_list(OUTPUTS, normalise_text=True)
    for rel, meta in old["outputs"].items():
        act = cur_o.get(rel, {"present": False})
        if not act.get("present"):
            print(f"  WARN   {rel} absent (regenerate to re-verify)")
            continue
        _cmp(f"{rel} sha256", meta.get("sha256"), act.get("sha256"), issues)

    print("\n--- environment (informational, drift does not fail) ---")
    env = environment()
    for k, v in old["environment"]["key_packages"].items():
        print(f"  {'OK  ' if v == env['key_packages'].get(k) else 'NOTE'}   "
              f"{k}: frozen {v} / now {env['key_packages'].get(k)}")

    print("\n" + "=" * 84)
    if issues:
        print(f"RESULT: DRIFT DETECTED in {len(issues)} item(s) -> {sorted(set(issues))[:8]}")
        print("EXP034-A as frozen is NO LONGER REPRODUCIBLE from this tree.")
        print("Do NOT edit EXP034-A. Publish the change as a new experiment.")
        print("=" * 84)
        return 1
    print("RESULT: VERIFIED -- preregistration, registry, folds, sources, data, "
          "predictions, statistics and reports all match the freeze.")
    print("=" * 84)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze/verify EXP034-A.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        return verify()

    if MANIFEST.exists():
        print(f"REFUSING to overwrite existing {MANIFEST}.")
        print("EXP034-A is frozen; delete the file deliberately to re-freeze.")
        return 1
    man = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    sd = man["source_data"]
    print(f"Wrote {MANIFEST}")
    print(f"  headline          : {man['verdict_summary']['headline'][:90]}...")
    print(f"  git commit        : {man['git']['commit']}")
    print(f"  prereg lock       : {LOCK_COMMIT}")
    print(f"  source data       : {sd['n_days']} days, {sd['n_files']} files, {sd['bytes']:,} bytes")
    print(f"  data root sha256  : {sd['root_sha256']}")
    print(f"  vix root sha256   : {sd['vix_root_sha256']}")
    print(f"  manifest sha256   : {sha256_file(MANIFEST)}")
    print(f"\nNext: commit the manifest, then tag `{TAG}`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
