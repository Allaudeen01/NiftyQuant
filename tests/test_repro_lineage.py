"""Tests for reproducibility metadata and experiment lineage."""

from nifty_quant.repro import capture_environment, config_hash, package_versions
from nifty_quant.research.experiment import ExperimentTracker


def test_capture_environment_fields():
    env = capture_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "packages" in env
    assert "numpy" in env["packages"]
    assert "pandas" in env["packages"]


def test_package_versions_handles_missing():
    versions = package_versions(("numpy", "definitely_not_a_real_pkg_xyz"))
    assert versions["definitely_not_a_real_pkg_xyz"] == "not installed"
    assert versions["numpy"] != "not installed"


def test_config_hash_is_order_independent_and_stable():
    a = config_hash({"x": 1, "y": 2})
    b = config_hash({"y": 2, "x": 1})
    assert a == b
    c = config_hash({"x": 1, "y": 3})
    assert a != c


def test_experiment_records_repro_metadata(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.log(
        strategy_name="EmaCross",
        strategy_version="1.0.0",
        parameters={"fast": 10, "slow": 20},
        metrics={"sharpe": 1.5},
        feature_version="v1",
        data_version="nifty-2025",
    )
    loaded = tracker.load(exp.id)
    assert loaded.environment is not None
    assert "python_version" in loaded.environment
    assert loaded.config_hash is not None


def test_experiment_lineage_chain(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    root = tracker.log(
        strategy_name="EMA", strategy_version="1.0.0",
        parameters={"fast": 10}, metrics={"sharpe": 1.0},
    )
    child = tracker.log(
        strategy_name="EMA", strategy_version="2.0.0",
        parameters={"fast": 12}, metrics={"sharpe": 1.2},
        parent_id=root.id,
    )
    grandchild = tracker.log(
        strategy_name="EMA", strategy_version="3.0.0",
        parameters={"fast": 14}, metrics={"sharpe": 1.4},
        parent_id=child.id,
    )

    assert [e.id for e in tracker.children(root.id)] == [child.id]
    chain = tracker.lineage(grandchild.id)
    assert [e.id for e in chain] == [root.id, child.id, grandchild.id]
