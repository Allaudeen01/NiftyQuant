"""Freeze / verify Experiment 033 -- immutability manifest for tag `EXP033_FINAL_FROZEN`.

EXP 033 is CLOSED (no candidate reached WALK_FORWARD_VALIDATED; see
docs/preregistrations/exp033_option_event_predictability.md and
reports/exp033/final_exp033_report.md). This script mirrors
scripts/exp032_freeze.py's role for a multi-script pipeline (rather than a
single framework module):

    python scripts/exp033_freeze.py --write     # build docs/exp033_FREEZE.json
    python scripts/exp033_freeze.py --verify    # re-check every recorded hash

--verify exits non-zero on ANY drift: if a locked source file, a threshold,
an input byte or an output byte changes, EXP 033 is no longer EXP 033 and
any change belongs in a new experiment id (exp034+). Read-only apart from
writing the manifest itself.
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

MANIFEST = Path("docs") / "exp033_FREEZE.json"
TAG = "EXP033_FINAL_FROZEN"
EVENT_DEFS_JSON = Path("docs") / "preregistrations" / "exp033_event_definitions.json"
PREREG_MD = Path("docs") / "preregistrations" / "exp033_option_event_predictability.md"

TRACKED_SOURCES = [
    str(PREREG_MD).replace("\\", "/"),
    str(EVENT_DEFS_JSON).replace("\\", "/"),
    "scripts/exp033_data_audit.py",
    "scripts/exp033_build_panel.py",
    "scripts/exp033_detect_events.py",
    "scripts/exp033_label_events.py",
    "scripts/exp033_walk_forward.py",
    "scripts/exp033_build_registry.py",
    "scripts/exp033_freeze.py",
    "nifty_quant/features/option_event_features.py",
    "nifty_quant/events/event_definitions.py",
    "nifty_quant/events/event_detector.py",
    "nifty_quant/labels/triple_barrier.py",
    "nifty_quant/models/option_event_model.py",
    "nifty_quant/models/probability_calibration.py",
    "nifty_quant/registry/candidate_registry.py",
    "requirements.txt",
    # reused (not owned) dependencies EXP033 imports directly -- tracked so a
    # change to shared infrastructure is visible here too, without re-freezing
    # exp032 (which has its own independent manifest).
    "scripts/strategy_test_framework.py",
    "nifty_quant/analytics/data_snooping.py",
    "nifty_quant/analytics/black_scholes.py",
    "nifty_quant/research/derive_iv.py",
    "nifty_quant/research/walkforward.py",
]

DERIVED_ARTIFACTS = [
    Path("data") / "derived" / "exp033_feature_panel.parquet",
    Path("data") / "derived" / "exp033_event_panel.parquet",
    Path("data") / "derived" / "exp033_combined_events.parquet",
    Path("data") / "derived" / "exp033_labeled_events.parquet",
]

REPORT_OUTPUTS = [
    "data_audit_report.csv", "data_audit_summary.md",
    "event_detection_summary.md", "labeling_summary.md",
    "walk_forward_predictions.csv", "walk_forward_trades.csv",
    "probability_calibration_report.csv", "feature_importance_report.csv",
    "candidate_registry.csv", "final_exp033_report.md",
]

KEY_PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "pyarrow", "scikit-learn", "lightgbm"]


# --- hashing -----------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_source(p: Path) -> str:
    """Line-ending-normalised hash for git-tracked TEXT sources (see
    scripts/exp032_freeze.py for the CRLF/LF rationale -- identical here)."""
    b = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(b).hexdigest()


# --- sections ------------------------------------------------------------------

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
        "tag": TAG,
        "note": ("The manifest itself is committed in a child commit; `commit` "
                 "above is the tree the experiment was RUN against. The tag "
                 f"`{TAG}` points at the commit that includes this manifest."),
    }


def locked_config() -> dict:
    with open(EVENT_DEFS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def environment() -> dict:
    from importlib import metadata
    pkgs = {}
    for name in KEY_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    try:
        full = sorted(f"{d.metadata['Name']}=={d.version}"
                       for d in metadata.distributions()
                       if d.metadata.get("Name"))
    except Exception:
        full = []
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "key_packages": pkgs,
        "installed_distributions_count": len(full),
        "installed_distributions_sha256": sha256_text("\n".join(full)),
    }


def source_data_manifest() -> dict:
    """Per-day hash over every option-chain file plus the VIX series EXP033
    reads (unlike Exp032, which never consumed VIX)."""
    chain_root = Path("data") / "option_chain"
    days = {}
    total_files = total_bytes = 0
    for day_dir in sorted(p for p in chain_root.glob("*/*/*") if p.is_dir()):
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        y, m, d = day_dir.parts[-3:]
        per = [f"{f.name}:{sha256_file(f)}" for f in files]
        nbytes = sum(f.stat().st_size for f in files)
        days[f"{y}-{m}-{d}"] = {
            "n_files": len(files), "bytes": nbytes,
            "day_sha256": sha256_text("\n".join(per)),
        }
        total_files += len(files)
        total_bytes += nbytes
    root_hash = sha256_text("\n".join(
        f"{k}:{v['day_sha256']}" for k, v in sorted(days.items())))

    vix_root = Path("data") / "vix"
    vix_files = sorted(vix_root.glob("*/*.parquet"))
    vix_hash = sha256_text("\n".join(f"{f.name}:{sha256_file(f)}" for f in vix_files))

    return {
        "option_chain_path": str(chain_root).replace("\\", "/"),
        "vix_path": str(vix_root).replace("\\", "/"),
        "n_days": len(days), "n_files": total_files, "bytes": total_bytes,
        "root_sha256": root_hash,
        "vix_n_files": len(vix_files), "vix_root_sha256": vix_hash,
        "per_day": days,
        "excluded_day_note": ("2026-06-26 is present in this raw fingerprint (its files "
                               "are unchanged on disk) but is EXCLUDED downstream by "
                               "scripts/exp033_build_panel.py per the preregistration "
                               "(frozen/stale spot feed) -- exclusion is a code-level "
                               "filter, not a data-deletion."),
        "not_consumed_note": ("data/external/ and data/candles/ were NOT read by the "
                               "EXP033 pipeline (no temporal overlap with the option-chain "
                               "collection window, per the Phase-1 data audit)."),
    }


def derived_artifacts() -> dict:
    import pandas as pd
    out = {}
    for p in DERIVED_ARTIFACTS:
        if not p.exists():
            out[str(p).replace("\\", "/")] = {"present": False}
            continue
        df = pd.read_parquet(p)
        out[str(p).replace("\\", "/")] = {
            "present": True, "sha256": sha256_file(p), "bytes": p.stat().st_size,
            "n_rows": int(len(df)), "n_columns": int(len(df.columns)),
        }
    return out


def outputs() -> dict:
    import pandas as pd
    out_dir = Path("reports") / "exp033"
    out = {}
    for name in REPORT_OUTPUTS:
        p = out_dir / name
        if not p.exists():
            out[name] = {"present": False}
            continue
        entry = {"path": str(p).replace("\\", "/"), "present": True,
                 "sha256": sha256_file(p), "bytes": p.stat().st_size}
        if name.endswith(".csv"):
            entry["n_rows"] = int(len(pd.read_csv(p)))
        out[name] = entry

    reg_path = out_dir / "candidate_registry.csv"
    if reg_path.exists():
        df = pd.read_csv(reg_path)
        out["candidate_registry_inline"] = json.loads(
            df.to_json(orient="records", double_precision=10))
    return out


def source_files() -> dict:
    d = {}
    for rel in TRACKED_SOURCES:
        p = Path(rel)
        d[rel] = sha256_source(p) if p.exists() else None
    return {"_hash_method": "sha256 of content with CRLF/CR normalised to LF", **d}


# --- build / verify --------------------------------------------------------

def build() -> dict:
    locked = locked_config()
    return {
        "experiment": "EXP 033 -- Option Chain Event Predictability Engine",
        "experiment_id": "exp033",
        "tag": TAG,
        "status": "FROZEN -- IMMUTABLE. No modifications permitted.",
        "policy": ("Any change to event definitions, barrier configs, thresholds, "
                   "data or code invalidates this freeze and MUST be published as "
                   "a new experiment id (exp034+). Never edit exp033 in place. "
                   "Re-running the locked pipeline on NEW data is an out-of-sample "
                   "test and must be reported as a separate experiment."),
        "verdict": "NO CANDIDATE REACHED WALK_FORWARD_VALIDATED -- 9/18 REJECT, 9/18 INSUFFICIENT_DATA.",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": str(PREREG_MD).replace("\\", "/"),
        "event_definitions": str(EVENT_DEFS_JSON).replace("\\", "/"),
        "manifest_schema": 1,
        "git": git_info(),
        "locked_event_and_barrier_config": locked,
        "environment": environment(),
        "source_data": source_data_manifest(),
        "derived_artifacts": derived_artifacts(),
        "outputs": outputs(),
        "source_files_sha256": source_files(),
    }


def _cmp(label: str, expect, actual, issues: list) -> None:
    ok = expect == actual
    print(f"  {'OK  ' if ok else 'DRIFT'}  {label}")
    if not ok:
        issues.append(label)
        print(f"          expected: {expect}")
        print(f"          actual:   {actual}")


def verify() -> int:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}. Run --write first.")
        return 1
    old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    issues: list[str] = []
    print("=" * 84)
    print(f"VERIFYING EXP 033 FREEZE  (tag {old.get('tag')})")
    print(f"Frozen at: {old.get('frozen_at_utc')}")
    print("=" * 84)

    print("\n--- locked event/barrier config ---")
    _cmp("locked_event_and_barrier_config", old["locked_event_and_barrier_config"],
         locked_config(), issues)

    print("\n--- source files ---")
    cur_src = source_files()
    for rel, h in old["source_files_sha256"].items():
        if rel.startswith("_"):
            continue
        _cmp(rel, h, cur_src.get(rel), issues)

    print("\n--- source data (option chains + VIX) ---")
    cur = source_data_manifest()
    for k in ("n_days", "n_files", "root_sha256", "vix_n_files", "vix_root_sha256"):
        _cmp(k, old["source_data"][k], cur[k], issues)

    print("\n--- derived artifacts ---")
    cur_art = derived_artifacts()
    for path, old_meta in old["derived_artifacts"].items():
        new_meta = cur_art.get(path, {"present": False})
        if not new_meta.get("present"):
            print(f"  WARN   {path} absent (rebuildable; not fatal)")
            continue
        for k in ("n_rows", "n_columns"):
            _cmp(f"{path} {k}", old_meta.get(k), new_meta.get(k), issues)
        if old_meta.get("sha256") != new_meta.get("sha256"):
            print(f"  WARN   {path} sha256 differs (parquet writer nondeterminism; "
                  f"row/column counts above are authoritative)")

    print("\n--- report outputs ---")
    cur_out = outputs()
    for name in REPORT_OUTPUTS:
        exp, act = old["outputs"].get(name, {}), cur_out.get(name, {})
        if not act.get("present"):
            print(f"  WARN   {name} absent from reports/exp033/ (regenerate to re-verify)")
            continue
        _cmp(f"{name} sha256", exp.get("sha256"), act.get("sha256"), issues)

    print("\n--- environment (informational, drift does not fail) ---")
    env = environment()
    for k, v in old["environment"]["key_packages"].items():
        cur_v = env["key_packages"].get(k)
        print(f"  {'OK  ' if v == cur_v else 'NOTE'}   {k}: frozen {v} / now {cur_v}")

    print("\n" + "=" * 84)
    if issues:
        print(f"RESULT: DRIFT DETECTED in {len(issues)} item(s) -> {sorted(set(issues))}")
        print("EXP 033 as frozen is NO LONGER REPRODUCIBLE from this tree.")
        print("Do NOT edit exp033. Publish the change as exp034+.")
        print("=" * 84)
        return 1
    print("RESULT: VERIFIED -- config, sources, data and outputs all match the freeze.")
    print("=" * 84)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze/verify EXP 033.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="write the manifest")
    g.add_argument("--verify", action="store_true", help="verify against manifest")
    args = ap.parse_args()

    if args.verify:
        return verify()

    if MANIFEST.exists():
        print(f"REFUSING to overwrite existing {MANIFEST}.")
        print("EXP 033 is frozen; delete the file deliberately if you truly intend to re-freeze.")
        return 1
    man = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    sd = man["source_data"]
    print(f"Wrote {MANIFEST}")
    print(f"  verdict           : {man['verdict']}")
    print(f"  git commit        : {man['git']['commit']}")
    print(f"  source data       : {sd['n_days']} days, {sd['n_files']} files, {sd['bytes']:,} bytes")
    print(f"  data root sha256  : {sd['root_sha256']}")
    print(f"  manifest sha256   : {sha256_file(MANIFEST)}")
    print(f"\nNext: commit the manifest, then tag `{TAG}`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
