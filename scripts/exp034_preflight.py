"""EXP034 mandatory preflight consistency gate.

The locked preregistration (docs/preregistrations/exp034_option_information_content.md,
section "Pre-run consistency check") requires that NO EXP034 execution may begin
unless the locked configuration and the implementation agree exactly. This
script is that gate: it exits NON-ZERO on any mismatch.

    python scripts/exp034_preflight.py                 # config-only checks
    python scripts/exp034_preflight.py --panel <path>  # also check a built panel

Rationale: EXP033's freeze tooling exists precisely to catch this class of
drift, and REV-1 of the EXP034 preregistration shipped a 14-vs-15 feature
discrepancy between its prose and its JSON. This gate makes that class of
error impossible to run with.

Read-only. Never writes, never modifies EXP033.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# The commit at which the preregistration + config became binding.
LOCK_COMMIT = "18a178e94dcb2df985c304d3a037de50407acbba"
CONFIG_PATH = "docs/preregistrations/exp034_config.json"

REQUIRED_FEATURE_KEYS = [
    "feature_id", "feature_name", "exact_formula", "input_fields",
    "lookback_snapshots", "normalization", "missing_value_rule",
    "ablation_group", "included_in_model_matrix",
]
REQUIRED_AUDIT_COLUMNS = [
    "feature_timestamp", "prediction_start_timestamp",
    "prediction_end_timestamp", "elapsed_seconds_t_to_t1",
]
EXPECTED_FEATURE_IDS = [f"F{i:02d}" for i in range(1, 16)]
EXPECTED_MATRIX = {"MODEL_1": 30, "MODEL_2": 35, "MODEL_3": 51}
EXPECTED_N_TARGETS = 4
EXPECTED_N_HYPOTHESES = 4


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'OK   ' if ok else 'FAIL '}  {label}")
        if not ok:
            if detail:
                print(f"           {detail}")
            self.failures.append(label)
        return ok


def _norm_sha256(data: bytes) -> str:
    return hashlib.sha256(
        data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def check_config_hash(gate: Gate, config_path: Path) -> None:
    """The working config must be byte-identical to the version locked at LOCK_COMMIT."""
    try:
        locked = subprocess.run(
            ["git", "show", f"{LOCK_COMMIT}:{CONFIG_PATH}"],
            capture_output=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        gate.check("config hash matches lock commit", False, f"cannot read locked blob: {exc}")
        return
    current = config_path.read_bytes()
    locked_h, current_h = _norm_sha256(locked), _norm_sha256(current)
    gate.check(
        f"config matches the version locked at {LOCK_COMMIT[:7]}",
        locked_h == current_h,
        f"locked={locked_h[:16]}... current={current_h[:16]}...",
    )


def check_features(gate: Gate, cfg: dict) -> None:
    reg = cfg.get("feature_registry", {})
    feats = reg.get("features", [])
    ids = [f.get("feature_id") for f in feats]

    gate.check("chain feature count == 15", len(feats) == 15, f"got {len(feats)}")
    gate.check("declared n_chain_features == 15", reg.get("n_chain_features") == 15,
               f"got {reg.get('n_chain_features')}")
    gate.check("feature IDs are exactly F01-F15, each once",
               sorted(ids) == EXPECTED_FEATURE_IDS,
               f"got {sorted(i for i in ids if i)}")
    gate.check("auxiliary indicator count == 1",
               len(reg.get("auxiliary_indicators", [])) == 1)

    missing = [
        f"{f.get('feature_id')}:{k}"
        for f in feats for k in REQUIRED_FEATURE_KEYS if k not in f
    ]
    gate.check("every feature carries all required registry keys", not missing,
               f"missing: {missing[:5]}")

    groups = reg.get("ablation_groups", {})
    flat = [x for g in groups.values() for x in g]
    gate.check("ablation groups partition F01-F15 exactly",
               sorted(flat) == EXPECTED_FEATURE_IDS and len(flat) == len(set(flat)))
    # Cross-check against the features ACTUALLY present, not just the expected
    # ID set -- otherwise dropping a feature while leaving the group map intact
    # would slip through (observed during this gate's own failure demo).
    gate.check("ablation groups match the features actually registered",
               sorted(flat) == sorted(i for i in ids if i),
               f"groups={sorted(flat)} vs registered={sorted(i for i in ids if i)}")
    tag_ok = all(
        f.get("ablation_group") == next(
            (g for g, v in groups.items() if f.get("feature_id") in v), None)
        for f in feats
    )
    gate.check("each feature's ablation_group tag matches the group map", tag_ok)
    gate.check("all 15 features are included_in_model_matrix",
               all(f.get("included_in_model_matrix") is True for f in feats))


def check_matrices(gate: Gate, cfg: dict) -> None:
    mm = cfg.get("model_matrices", {})
    m1_list = mm.get("MODEL_1_column_list", [])
    add2 = mm.get("MODEL_2_added_columns", [])
    add3 = mm.get("MODEL_3_added_columns", [])

    gate.check("MODEL_1 declared == 30", mm.get("MODEL_1_columns") == EXPECTED_MATRIX["MODEL_1"])
    gate.check("MODEL_1 enumerated list length == 30", len(m1_list) == 30, f"got {len(m1_list)}")
    gate.check("MODEL_1 column names unique", len(set(m1_list)) == len(m1_list))
    gate.check("MODEL_2 == MODEL_1 + added == 35",
               mm.get("MODEL_1_columns", 0) + len(add2) == mm.get("MODEL_2_columns") == 35)
    gate.check("MODEL_3 == MODEL_2 + added == 51",
               mm.get("MODEL_2_columns", 0) + len(add3) == mm.get("MODEL_3_columns") == 51)

    # Declared nesting: MODEL_1 subset of MODEL_2 subset of MODEL_3.
    cols1 = list(m1_list)
    cols2 = cols1 + list(add2)
    cols3 = cols2 + list(add3)
    gate.check("declared nesting MODEL_1 subset MODEL_2", set(cols1).issubset(set(cols2)))
    gate.check("declared nesting MODEL_2 subset MODEL_3", set(cols2).issubset(set(cols3)))
    gate.check("no duplicate columns across the full MODEL_3 matrix",
               len(set(cols3)) == len(cols3))


def check_hypotheses(gate: Gate, cfg: dict) -> None:
    tgt = cfg.get("targets", {})
    mt = cfg.get("multiple_testing", {})
    primary = tgt.get("primary", {})
    gate.check("primary target count == 4",
               tgt.get("n_primary_targets") == EXPECTED_N_TARGETS == len(primary),
               f"declared={tgt.get('n_primary_targets')} listed={len(primary)}")
    gate.check("primary target IDs are T1a/T1b/T2a/T2b",
               sorted(primary) == ["T1a", "T1b", "T2a", "T2b"])
    gate.check("primary hypothesis count == 4",
               mt.get("n_primary_hypotheses") == EXPECTED_N_HYPOTHESES == len(mt.get("primary_family", [])))


def check_panel(gate: Gate, panel_path: Path) -> None:
    import pandas as pd

    df = pd.read_parquet(panel_path)
    missing = [c for c in REQUIRED_AUDIT_COLUMNS if c not in df.columns]
    gate.check("mandatory audit columns present", not missing, f"missing: {missing}")
    if missing:
        return

    # Timing invariants are asserted over RETAINED rows -- excluded rows are
    # precisely the ones allowed to violate a cap (that is why they carry an
    # exclusion_reason).
    ret = df[df["retained"]] if "retained" in df.columns else df
    gate.check("panel carries a `retained` flag", "retained" in df.columns)

    start = pd.to_datetime(ret["prediction_start_timestamp"])
    feat = pd.to_datetime(ret["feature_timestamp"])
    end = pd.to_datetime(ret["prediction_end_timestamp"])
    bad = int((start <= feat).sum())
    gate.check("prediction_start_timestamp > feature_timestamp for EVERY retained row",
               bad == 0, f"{bad} violating rows")
    bad_end = int((end <= start).sum())
    gate.check("prediction_end_timestamp > prediction_start_timestamp for every retained row",
               bad_end == 0, f"{bad_end} violating rows")
    over = int((ret["elapsed_seconds_t_to_t1"] > 300).sum())
    gate.check("no retained row exceeds the 300s t->t+1 cap", over == 0, f"{over} rows over cap")
    # Excluded rows must each carry a reason -- no silent drops.
    if "exclusion_reason" in df.columns:
        silent = int(((~df["retained"]) & (df["exclusion_reason"].fillna("") == "")).sum())
        gate.check("every excluded row carries an exclusion_reason", silent == 0,
                   f"{silent} silently dropped rows")


def check_feature_panel(gate: Gate, cfg: dict, path: Path) -> None:
    """Validate the REALIZED model matrices, not just the declared column lists."""
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(path)
    reg = cfg["feature_registry"]
    chain = [f["feature_name"] for f in reg["features"]]
    mm = cfg["model_matrices"]
    m1 = list(mm["MODEL_1_column_list"])
    m2 = m1 + list(mm["MODEL_2_added_columns"])
    m3 = m2 + chain + ["iv_available"]

    for name, cols, expected in (("MODEL_1", m1, mm["MODEL_1_columns"]),
                                  ("MODEL_2", m2, mm["MODEL_2_columns"]),
                                  ("MODEL_3", m3, mm["MODEL_3_columns"])):
        present = [c for c in cols if c in df.columns]
        gate.check(f"realized {name} has all {expected} locked columns",
                   len(present) == expected == len(cols),
                   f"present {len(present)} of {len(cols)}, locked {expected}; "
                   f"missing={[c for c in cols if c not in df.columns][:6]}")

    gate.check("realized nesting MODEL_1 subset MODEL_2 subset MODEL_3",
               set(m1).issubset(set(m2)) and set(m2).issubset(set(m3)))
    gate.check("all 15 locked chain features present as columns",
               all(c in df.columns for c in chain),
               f"missing={[c for c in chain if c not in df.columns]}")

    num = df[[c for c in m3 if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    gate.check("no infinite values in the MODEL_3 matrix",
               int(np.isinf(num.to_numpy()).sum()) == 0)
    degenerate = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    gate.check("no constant/degenerate columns in the MODEL_3 matrix",
               not degenerate, f"degenerate={degenerate}")

    if {"prediction_start_timestamp", "feature_timestamp"}.issubset(df.columns):
        ok = int((pd.to_datetime(df["prediction_start_timestamp"])
                  > pd.to_datetime(df["feature_timestamp"])).sum())
        gate.check("timing invariant survives the feature join",
                   ok == len(df), f"{len(df) - ok} violating rows")


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP034 preflight consistency gate.")
    ap.add_argument("--config", default=CONFIG_PATH,
                    help="config to validate (default: the locked path)")
    ap.add_argument("--panel", default=None,
                    help="optional built observation panel to validate")
    ap.add_argument("--features", default=None,
                    help="optional built feature panel; validates REALIZED matrices")
    args = ap.parse_args()

    config_path = Path(args.config)
    print("=" * 78)
    print("EXP034 PREFLIGHT CONSISTENCY GATE")
    print(f"lock commit : {LOCK_COMMIT}")
    print(f"config      : {config_path}")
    print("=" * 78)

    if not config_path.exists():
        print(f"\nFATAL: config not found at {config_path}")
        return 2
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    gate = Gate()
    print("\n--- locked config integrity ---")
    check_config_hash(gate, config_path)
    print("\n--- feature registry ---")
    check_features(gate, cfg)
    print("\n--- model matrices and nesting ---")
    check_matrices(gate, cfg)
    print("\n--- targets and hypotheses ---")
    check_hypotheses(gate, cfg)

    if args.panel:
        print("\n--- built observation panel ---")
        p = Path(args.panel)
        if not p.exists():
            gate.check("panel exists", False, str(p))
        else:
            check_panel(gate, p)

    if args.features:
        print("\n--- built feature panel (realized matrices) ---")
        fp = Path(args.features)
        if not fp.exists():
            gate.check("feature panel exists", False, str(fp))
        else:
            check_feature_panel(gate, cfg, fp)

    print("\n" + "=" * 78)
    if gate.failures:
        print(f"PREFLIGHT FAILED: {len(gate.failures)} check(s) -> {gate.failures}")
        print("EXP034 MUST NOT RUN. Locked config and implementation disagree.")
        print("=" * 78)
        return 1
    print("PREFLIGHT PASSED: locked configuration and implementation agree.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
