"""Run the Daily Research Pipeline for one trading session.

Turns a day's collected option-chain and candle data into a concise, durable
research report and maintains an evidence-based research journal. This is
read-only analysis: it never places, modifies, or simulates a trade.

    # Most recent collected session, deterministic rule-based ideas (default)
    python scripts/daily_research.py

    # An explicit Target_Session
    python scripts/daily_research.py --session 2025-06-19

    # Tighten the quality gate and require a larger sample before significance
    python scripts/daily_research.py --quality-threshold 80 --min-sample 30

The command is a thin wiring layer: it parses arguments, builds a
``PipelineConfig``, constructs the frozen ``nifty_quant`` components the
pipeline orchestrates (warehouse storage, feature store, research journal, and
a deterministic idea generator), runs :meth:`PipelineOrchestrator.run`, prints a
concise summary, and returns the run's exit code.

With ``--no-ai`` (the default) the whole run is deterministic: the rule-based
idea generator is the primary hypothesis source. ``--ai`` is an opt-in
placeholder -- no AI backend is wired in this build, so it currently falls back
to the same deterministic rule-based generator.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.features.store import ParquetFeatureStore
from nifty_quant.log import get_logger
from nifty_quant.research.journal import ResearchJournal
from nifty_quant.research.pipeline.ideas import RuleBasedIdeaGenerator
from nifty_quant.research.pipeline.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
)
from nifty_quant.research.pipeline.quality import QualityConfig

_log = get_logger("scripts.daily_research")

# Returned by main() when argument values are malformed (argparse uses 2 for its
# own parse errors; we mirror that for an invalid --session date value).
_EXIT_USAGE = 2


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the daily research pipeline for one session (read-only).",
    )
    p.add_argument(
        "--session",
        default=None,
        help="Explicit Target_Session as YYYY-MM-DD "
        "(default: the most recent collected session).",
    )
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--feature-dir", default="data")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--journal", default="reports/research_journal.jsonl")
    p.add_argument("--roadmap", default="ROADMAP.md")

    ai = p.add_mutually_exclusive_group()
    ai.add_argument(
        "--ai",
        dest="use_ai",
        action="store_true",
        help="Opt-in AI idea enhancement (placeholder: falls back to rule-based).",
    )
    ai.add_argument(
        "--no-ai",
        dest="use_ai",
        action="store_false",
        help="Disable AI ideas; use the deterministic rule-based generator (default).",
    )
    p.set_defaults(use_ai=False)

    p.add_argument(
        "--min-sample",
        type=int,
        default=20,
        help="Minimum collected sessions before significance language is allowed.",
    )
    p.add_argument(
        "--quality-threshold",
        type=int,
        default=70,
        help="Minimum Quality_Score (0..100) for a PASS verdict on the data gate.",
    )
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> "PipelineConfig | None":
    """Build a :class:`PipelineConfig` from parsed args.

    Returns ``None`` when ``--session`` is supplied but malformed, signalling the
    caller to return the usage exit code (Req 1.5).
    """
    session: date | None = None
    if args.session is not None:
        try:
            session = date.fromisoformat(args.session)
        except ValueError:
            _log.error("invalid --session %r: expected YYYY-MM-DD", args.session)
            return None

    return PipelineConfig(
        underlying=args.underlying,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        report_dir=args.report_dir,
        journal_path=args.journal,
        roadmap_path=args.roadmap,
        session=session,
        use_ai=args.use_ai,
        min_sample_size=args.min_sample,
        # QualityConfig is frozen: construct it with the requested threshold.
        quality=QualityConfig(pass_threshold=args.quality_threshold),
    )


def main(argv: "list[str] | None" = None) -> int:
    """Parse args, wire the pipeline, run it, and return the exit code."""
    args = parse_args(argv)

    config = _build_config(args)
    if config is None:
        return _EXIT_USAGE

    # Construct the frozen components the pipeline orchestrates (Req 17.2, 17.3).
    storage = ParquetStorage(config.data_dir)
    feature_store = ParquetFeatureStore(config.feature_dir)
    journal = ResearchJournal(config.journal_path)
    # Rule-based generation is the primary, deterministic source (Req 15.1). The
    # --ai flag is an opt-in placeholder; no AI backend is wired in this build,
    # so it currently falls back to the same rule-based generator.
    idea_generator = RuleBasedIdeaGenerator()
    if config.use_ai:
        _log.warning(
            "--ai requested but no AI backend is configured; "
            "falling back to the deterministic rule-based idea generator."
        )

    orchestrator = PipelineOrchestrator(
        config, storage, feature_store, journal, idea_generator
    )
    result = orchestrator.run()

    print("=" * 56)
    print(f"DAILY RESEARCH  {config.underlying} {config.timeframe}")
    print(f"Target session  : {result.target_session}")
    print(f"Collected days  : {result.collected_session_count}")
    print(
        "Quality score   : "
        + (str(result.quality_score) if result.quality_score is not None else "n/a")
    )
    print(
        "Confidence score: "
        + (
            str(result.confidence_score)
            if result.confidence_score is not None
            else "n/a"
        )
    )
    if result.exit_code == 0:
        print(f"Report          : {result.report_path}")
    else:
        print(f"Failing stage   : {result.failing_stage}")
        print(f"Detail          : {result.error_detail}")
    print(f"Exit code       : {result.exit_code}")
    print("=" * 56)

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
