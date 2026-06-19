"""Experiment tracking.

Records every backtest run with enough provenance to reproduce and compare it
later: strategy name/version, parameters, feature version, a caller-supplied
data version, the current git commit, the resulting metrics, the market regime,
and a timestamp. Runs are stored as JSON files plus an append-only index.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nifty_quant.log import get_logger

_log = get_logger("research.experiment")


def git_commit(short: bool = True) -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return None


@dataclass
class Experiment:
    id: str
    timestamp: str
    strategy_name: str
    strategy_version: str
    parameters: dict
    feature_version: str | None
    data_version: str | None
    git_commit: str | None
    metrics: dict
    regime: dict | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    parent_id: str | None = None
    environment: dict | None = None
    config_hash: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentTracker:
    """Persists and queries experiment records under ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.dir = self.root / "experiments"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

    def log(
        self,
        *,
        strategy_name: str,
        strategy_version: str,
        parameters: dict,
        metrics: dict,
        feature_version: str | None = None,
        data_version: str | None = None,
        regime: dict | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        parent_id: str | None = None,
        capture_git: bool = True,
        capture_environment: bool = True,
    ) -> Experiment:
        from nifty_quant.repro import capture_environment as _capture_env
        from nifty_quant.repro import config_hash as _config_hash

        now = datetime.now(timezone.utc)
        environment = _capture_env() if capture_environment else None
        cfg_hash = _config_hash(
            {
                "strategy_name": strategy_name,
                "strategy_version": strategy_version,
                "parameters": parameters,
                "feature_version": feature_version,
                "data_version": data_version,
            }
        )
        exp = Experiment(
            id=f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
            timestamp=now.isoformat(),
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            parameters=parameters,
            feature_version=feature_version,
            data_version=data_version,
            git_commit=git_commit() if capture_git else None,
            metrics=metrics,
            regime=regime,
            tags=tags or [],
            notes=notes,
            parent_id=parent_id,
            environment=environment,
            config_hash=cfg_hash,
        )
        self._write(exp)
        _log.event(
            "experiment_logged",
            id=exp.id,
            strategy=strategy_name,
            parent_id=parent_id,
            git_commit=exp.git_commit,
        )
        return exp

    def _write(self, exp: Experiment) -> None:
        payload = json.dumps(exp.to_dict(), default=str)
        (self.dir / f"{exp.id}.json").write_text(payload, encoding="utf-8")
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    def list_experiments(self) -> list[Experiment]:
        if not self.index_path.exists():
            return []
        out: list[Experiment] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Experiment(**json.loads(line)))
        return out

    def load(self, experiment_id: str) -> Experiment:
        path = self.dir / f"{experiment_id}.json"
        return Experiment(**json.loads(path.read_text(encoding="utf-8")))

    def children(self, experiment_id: str) -> list[Experiment]:
        """Direct descendants of an experiment."""
        return [e for e in self.list_experiments() if e.parent_id == experiment_id]

    def lineage(self, experiment_id: str) -> list[Experiment]:
        """Ancestry chain from the root down to ``experiment_id`` (inclusive)."""
        by_id = {e.id: e for e in self.list_experiments()}
        chain: list[Experiment] = []
        current = by_id.get(experiment_id)
        seen: set[str] = set()
        while current is not None and current.id not in seen:
            chain.append(current)
            seen.add(current.id)
            current = by_id.get(current.parent_id) if current.parent_id else None
        chain.reverse()
        return chain

    def to_frame(self) -> pd.DataFrame:
        """Flatten experiments (params + metrics) into a comparison table."""
        rows = []
        for exp in self.list_experiments():
            row = {
                "id": exp.id,
                "timestamp": exp.timestamp,
                "strategy": exp.strategy_name,
                "version": exp.strategy_version,
                "feature_version": exp.feature_version,
                "git_commit": exp.git_commit,
            }
            row.update({f"param.{k}": v for k, v in exp.parameters.items()})
            row.update({f"metric.{k}": v for k, v in exp.metrics.items()})
            if exp.regime:
                row["regime.trend"] = exp.regime.get("trend")
                row["regime.volatility"] = exp.regime.get("volatility")
            rows.append(row)
        return pd.DataFrame(rows)
