"""Research platform: regime tagging, sweeps, walk-forward, experiment tracking.

These tools turn the backtest engine into a reproducible research workflow:
test many parameter sets, validate out-of-sample, tag the market regime, and
record every run (params, metrics, feature version, git commit) so results are
comparable months later.
"""

from nifty_quant.research.regime import Regime, RegimeConfig, classify_regime
from nifty_quant.research.sweep import (
    SweepResult,
    SweepReport,
    expand_grid,
    run_sweep,
)
from nifty_quant.research.walkforward import (
    WalkForwardWindow,
    WalkForwardResult,
    generate_windows,
    run_walk_forward,
)
from nifty_quant.research.experiment import (
    Experiment,
    ExperimentTracker,
    git_commit,
)

__all__ = [
    "Regime",
    "RegimeConfig",
    "classify_regime",
    "SweepResult",
    "SweepReport",
    "expand_grid",
    "run_sweep",
    "WalkForwardWindow",
    "WalkForwardResult",
    "generate_windows",
    "run_walk_forward",
    "Experiment",
    "ExperimentTracker",
    "git_commit",
]
