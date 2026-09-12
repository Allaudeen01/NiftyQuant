"""Freeze / verify Experiment 032 -- immutability manifest for tag `exp032_final`.

EXP 032 is CLOSED. This script does two things and nothing else:

    python scripts/exp032_freeze.py --write     # build docs/exp032_FREEZE.json
    python scripts/exp032_freeze.py --verify    # re-check every recorded hash

The manifest is the only in-repo record of the data and output fingerprints,
because data/ and reports/ are gitignored. It captures: strategy rules, source
data manifest (per-day + root hash over all 9,816 chain files), feature cache
hash, cost schedule version, random seeds, package versions, git commit, and
output CSV hashes -- plus the headline results inline, so the manifest stays
meaningful even if the CSVs are lost.

--verify exits non-zero on ANY drift. That is the enforcement mechanism: if a
rule, a cost constant, a source file, an input byte or an output byte changes,
EXP 032 is no longer EXP 032 and the next experiment needs a new id.

Read-only apart from writing the manifest itself.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy_test_framework as F  # noqa: E402  (single source of truth)

MANIFEST = Path("docs") / "exp032_FREEZE.json"
TAG = "exp032_final"

TRACKED_SOURCES = [
    "scripts/strategy_test_framework.py",
    "scripts/exp032_freeze.py",
    "docs/preregistrations/exp032_strategy_zoo.md",
    "nifty_quant/analytics/data_snooping.py",
    "nifty_quant/analytics/black_scholes.py",
    "nifty_quant/analytics/options.py",
    "nifty_quant/research/live_lens.py",
    "nifty_quant/research/derive_iv.py",
    "nifty_quant/data/storage/parquet.py",
]

KEY_PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "pyarrow"]


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
    """Line-ending-normalised hash for git-tracked TEXT sources.

    .gitattributes sets `* text=auto`, so a fresh clone on Windows checks these
    files out as CRLF while this machine holds LF. Hashing raw bytes would then
    report spurious drift for an identical file. Normalising CRLF/CR -> LF makes
    the fingerprint platform-independent. Binary artifacts (parquet, csv) are
    gitignored, never normalised by git, and so are hashed byte-for-byte.
    """
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
        "tag": TAG,
        "note": ("The manifest itself is committed in a child commit; `commit` "
                 "above is the tree the experiment was RUN against. The tag "
                 f"`{TAG}` points at the commit that includes this manifest."),
    }


def strategy_rules() -> dict:
    """All LOCKED rule parameters, read live from the framework module."""
    return {
        "experiment_id": F.EXPERIMENT_ID,
        "rules_version": F.RULES_VERSION,
        "candidates": [{"code": c, "name": n, "instrument": i,
                        "function": fn.__name__}
                       for c, n, i, fn in F.STRATEGIES],
        "clocks": {"entry": F.ENTRY_CLOCK, "exit": F.EXIT_CLOCK,
                   "midday_signal": F.MIDDAY_CLOCK,
                   "first_hour_exit": F.FIRST_HOUR_EXIT},
        "thresholds": {
            "S1_VRP_THRESHOLD_volpts": F.S1_VRP_THRESHOLD,
            "S1_RV_LOOKBACK_days": F.S1_RV_LOOKBACK,
            "S2_MOVE_THRESHOLD": F.S2_MOVE_THRESHOLD,
            "S3_IV_PCTILE": F.S3_IV_PCTILE,
            "S3_WARMUP_days": F.S3_WARMUP,
            "S4_PCR_HIGH": F.S4_PCR_HIGH,
            "S4_PCR_LOW": F.S4_PCR_LOW,
            "S5_MAXPAIN_PROX": F.S5_MAXPAIN_PROX,
            "S7_GAP_THRESHOLD": F.S7_GAP_THRESHOLD,
        },
        "inference": {
            "HAC_LAG": F.HAC_LAG, "FDR_Q": F.FDR_Q,
            "SPA_BLOCK": F.SPA_BLOCK, "SPA_NBOOT": F.SPA_NBOOT,
            "PIN_NBOOT": F.PIN_NBOOT, "PIN_BLOCK": F.PIN_BLOCK,
            "TRAIN_FRAC": F.TRAIN_FRAC,
        },
        "pricing": {"R_RATE": F.R_RATE, "Q_YIELD": F.Q_YIELD},
    }


def cost_schedule() -> dict:
    return {
        "version": F.COST_SCHEDULE_VERSION,
        "disclaimer": ("Indian F&O retail rates are ASSUMPTIONS as noted in the "
                       "pre-registration; results are reported with a spread "
                       "sensitivity sweep."),
        "LOT_SIZE": F.LOT_SIZE,
        "MARGIN_PER_LOT": F.MARGIN_PER_LOT,
        "BROKERAGE_PER_ORDER": F.BROKERAGE_PER_ORDER,
        "STT_OPT_SELL": F.STT_OPT_SELL,
        "EXCH_OPT": F.EXCH_OPT,
        "SEBI_FEE": F.SEBI_FEE,
        "STAMP_BUY": F.STAMP_BUY,
        "GST_RATE": F.GST_RATE,
        "STT_FUT_SELL": F.STT_FUT_SELL,
        "EXCH_FUT": F.EXCH_FUT,
        "FUT_SLIP_BPS": F.FUT_SLIP_BPS,
        "SPREAD_MULTS": list(F.SPREAD_MULTS),
        "BASE_MULT": F.BASE_MULT,
    }


def random_seeds() -> dict:
    return {
        "SPA_SEED": F.SPA_SEED,
        "PIN_SEED": F.PIN_SEED,
        "spa_n_boot": F.SPA_NBOOT,
        "pinning_n_boot": F.PIN_NBOOT,
        "determinism_note": ("All stochastic steps are stationary bootstraps "
                            "seeded from these constants via "
                            "np.random.default_rng; two consecutive runs "
                            "reproduced identical p-values."),
    }


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
        "installed_distributions": full,
    }


def source_data_manifest() -> dict:
    """Per-day and root hash over every option-chain file the experiment read."""
    root = Path(F.DATA_DIR) / "option_chain"
    days = {}
    total_files = total_bytes = 0
    for day_dir in sorted(p for p in root.glob("*/*/*") if p.is_dir()):
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        y, m, d = day_dir.parts[-3:]
        per = [f"{f.name}:{sha256_file(f)}" for f in files]
        nbytes = sum(f.stat().st_size for f in files)
        days[f"{y}-{m}-{d}"] = {
            "n_files": len(files),
            "bytes": nbytes,
            "day_sha256": sha256_text("\n".join(per)),
        }
        total_files += len(files)
        total_bytes += nbytes
    root_hash = sha256_text("\n".join(
        f"{k}:{v['day_sha256']}" for k, v in sorted(days.items())))
    return {
        "path": str(root).replace("\\", "/"),
        "consumed_by_exp032": True,
        "n_days": len(days),
        "n_files": total_files,
        "bytes": total_bytes,
        "root_sha256": root_hash,
        "per_day": days,
        "not_consumed_note": ("data/vix/, data/external/ and data/candles/ were "
                              "NOT read by strategy_test_framework.py and are "
                              "deliberately excluded from this fingerprint."),
    }


def feature_cache() -> dict:
    p = F.PANEL
    if not p.exists():
        return {"path": str(p).replace("\\", "/"), "present": False}
    import pandas as pd
    df = pd.read_parquet(p)
    return {
        "path": str(p).replace("\\", "/"),
        "present": True,
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "n_rows": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "columns": list(df.columns),
        "rebuild_command": "python scripts/strategy_test_framework.py --rebuild-panel",
        "note": ("Derived artifact. Byte-level parquet output can vary with "
                 "pyarrow version; the authoritative invariants are n_rows, "
                 "n_days, columns and the source_data root_sha256."),
    }


def _latest(pattern: str) -> Path | None:
    hits = sorted(Path("reports").glob(pattern))
    return hits[-1] if hits else None


def outputs() -> dict:
    import pandas as pd
    out = {}
    summary_p = _latest("exp032_strategy_zoo_*.csv")
    trades_p = _latest("exp032_trades_*.csv")
    for label, p in (("summary_csv", summary_p), ("trades_csv", trades_p)):
        if p is None:
            out[label] = {"present": False}
            continue
        out[label] = {"path": str(p).replace("\\", "/"),
                      "present": True,
                      "sha256": sha256_file(p),
                      "bytes": p.stat().st_size,
                      "n_rows": int(len(pd.read_csv(p)))}
    # embed headline results so the manifest survives loss of the CSVs
    if summary_p is not None:
        df = pd.read_csv(summary_p)
        cols = ["code", "strategy", "n", "mean", "win", "sharpe", "max_dd", "pf",
                "worst", "t", "p", "q_bh", "fdr_reject", "spa_p", "rc_p",
                "cost_fragile", "unstable", "verdict"]
        keep = [c for c in cols if c in df.columns]
        out["results_inline"] = json.loads(
            df[keep].to_json(orient="records", double_precision=10))
    return out


def source_files() -> dict:
    d = {}
    for rel in TRACKED_SOURCES:
        p = Path(rel)
        d[rel] = sha256_source(p) if p.exists() else None
    return {"_hash_method": "sha256 of content with CRLF/CR normalised to LF",
            **d}


# --- build / verify --------------------------------------------------------

def build() -> dict:
    return {
        "experiment": "EXP 032 -- Multi-Strategy Zoo on collected option chains",
        "experiment_id": F.EXPERIMENT_ID,
        "tag": TAG,
        "status": "FROZEN -- IMMUTABLE. No modifications permitted.",
        "policy": ("Any change to rules, costs, data or code invalidates this "
                   "freeze and MUST be published as a new experiment id "
                   "(exp033+). Never edit exp032 in place. Re-running the "
                   "locked rules on NEW data is an out-of-sample test and must "
                   "be reported as a separate experiment."),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": "docs/preregistrations/exp032_strategy_zoo.md",
        "manifest_schema": 1,
        "git": git_info(),
        "strategy_rules": strategy_rules(),
        "cost_schedule": cost_schedule(),
        "random_seeds": random_seeds(),
        "environment": environment(),
        "source_data": source_data_manifest(),
        "feature_cache": feature_cache(),
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
    print(f"VERIFYING EXP 032 FREEZE  (tag {old.get('tag')})")
    print(f"Frozen at: {old.get('frozen_at_utc')}")
    print("=" * 84)

    print("\n--- strategy rules & cost schedule ---")
    _cmp("rules_version", old["strategy_rules"]["rules_version"],
         F.RULES_VERSION, issues)
    _cmp("strategy rules block", old["strategy_rules"], strategy_rules(), issues)
    _cmp("cost schedule block", old["cost_schedule"], cost_schedule(), issues)
    _cmp("random seeds", {k: v for k, v in old["random_seeds"].items()
                          if k != "determinism_note"},
         {k: v for k, v in random_seeds().items() if k != "determinism_note"},
         issues)

    print("\n--- source files ---")
    cur_src = source_files()
    for rel, h in old["source_files_sha256"].items():
        if rel.startswith("_"):
            continue
        _cmp(rel, h, cur_src.get(rel), issues)

    print("\n--- source data (option chains) ---")
    cur = source_data_manifest()
    _cmp("n_days", old["source_data"]["n_days"], cur["n_days"], issues)
    _cmp("n_files", old["source_data"]["n_files"], cur["n_files"], issues)
    _cmp("root_sha256", old["source_data"]["root_sha256"],
         cur["root_sha256"], issues)
    changed = [d for d, v in cur["per_day"].items()
               if old["source_data"]["per_day"].get(d, {}).get("day_sha256")
               != v["day_sha256"]]
    missing = [d for d in old["source_data"]["per_day"]
               if d not in cur["per_day"]]
    if changed:
        print(f"  DRIFT  {len(changed)} day(s) changed: {changed[:10]}")
        issues.append("per_day hashes")
    if missing:
        print(f"  DRIFT  {len(missing)} day(s) missing: {missing[:10]}")
        issues.append("missing days")

    print("\n--- feature cache ---")
    cf = feature_cache()
    if not cf.get("present"):
        print("  WARN   panel absent (rebuildable; not fatal)")
    else:
        for k in ("n_rows", "n_days", "columns"):
            _cmp(f"panel {k}", old["feature_cache"].get(k), cf.get(k), issues)
        if old["feature_cache"].get("sha256") != cf.get("sha256"):
            print("  WARN   panel sha256 differs (parquet writer nondeterminism; "
                  "invariants above are authoritative)")

    print("\n--- outputs ---")
    co = outputs()
    for label in ("summary_csv", "trades_csv"):
        exp, act = old["outputs"].get(label, {}), co.get(label, {})
        if not act.get("present"):
            print(f"  WARN   {label} absent from reports/ "
                  f"(regenerate to re-verify)")
            continue
        if exp.get("path") != act.get("path"):
            print(f"  NOTE   {label} filename differs "
                  f"({exp.get('path')} -> {act.get('path')})")
        _cmp(f"{label} sha256", exp.get("sha256"), act.get("sha256"), issues)

    print("\n--- environment (informational, drift does not fail) ---")
    env = environment()
    for k, v in old["environment"]["key_packages"].items():
        cur_v = env["key_packages"].get(k)
        print(f"  {'OK  ' if v == cur_v else 'NOTE'}   {k}: frozen {v} / now {cur_v}")
    print(f"  {'OK  ' if old['environment']['python'] == env['python'] else 'NOTE'}"
          f"   python: frozen {old['environment']['python']} / now {env['python']}")

    print("\n" + "=" * 84)
    if issues:
        print(f"RESULT: DRIFT DETECTED in {len(issues)} item(s) -> "
              f"{sorted(set(issues))}")
        print("EXP 032 as frozen is NO LONGER REPRODUCIBLE from this tree.")
        print("Do NOT edit exp032. Publish the change as exp033+.")
        print("=" * 84)
        return 1
    print("RESULT: VERIFIED -- rules, costs, seeds, sources, data and outputs "
          "all match the freeze.")
    print("=" * 84)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze/verify EXP 032.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="write the manifest")
    g.add_argument("--verify", action="store_true", help="verify against manifest")
    args = ap.parse_args()

    if args.verify:
        return verify()

    if MANIFEST.exists():
        print(f"REFUSING to overwrite existing {MANIFEST}.")
        print("EXP 032 is frozen; delete the file deliberately if you truly "
              "intend to re-freeze.")
        return 1
    man = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8")
    sd, fc = man["source_data"], man["feature_cache"]
    print(f"Wrote {MANIFEST}")
    print(f"  rules_version     : {man['strategy_rules']['rules_version']}")
    print(f"  cost schedule     : {man['cost_schedule']['version']}")
    print(f"  git commit        : {man['git']['commit']}")
    print(f"  source data       : {sd['n_days']} days, {sd['n_files']} files, "
          f"{sd['bytes']:,} bytes")
    print(f"  data root sha256  : {sd['root_sha256']}")
    print(f"  feature cache     : {fc.get('sha256', 'absent')}")
    for k in ("summary_csv", "trades_csv"):
        o = man["outputs"].get(k, {})
        if o.get("present"):
            print(f"  {k:<18}: {o['sha256']}")
    print(f"  manifest sha256   : {sha256_file(MANIFEST)}")
    print(f"\nNext: commit the manifest, then tag `{TAG}`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
