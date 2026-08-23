"""Tests for walk-forward splits and a hermetic MLflow smoke run."""

import mlflow
import pandas as pd
import pytest

from dapm.data import synthetic_prices
from dapm.train import run_walk_forward, walk_forward_splits


def test_splits_are_causal_expanding_and_tile_the_end():
    idx = pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz="UTC")
    splits = walk_forward_splits(idx, n_folds=4, test_hours=24 * 14, min_train_hours=24 * 100)
    assert len(splits) == 4
    prev_end = None
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        assert train_idx.min() == idx[0]
        if prev_end is not None:
            assert test_idx.min() == prev_end + pd.Timedelta(hours=1)
        prev_end = test_idx.max()
    assert prev_end == idx.max()


def test_splits_reject_insufficient_history():
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        walk_forward_splits(idx, n_folds=4, test_hours=24 * 14, min_train_hours=24 * 100)


def test_walk_forward_smoke_logs_to_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    prices = synthetic_prices("2024-01-01", "2024-07-01", seed=0)
    summary = run_walk_forward(
        prices,
        model_params={"max_iter": 20},
        n_folds=2,
        test_hours=24 * 7,
        min_train_hours=24 * 60,
        experiment="smoke",
    )
    assert len(summary) == 2
    assert summary["mae"].notna().all()
    runs = mlflow.search_runs(experiment_names=["smoke"])
    assert len(runs) == 3  # one parent + two fold children


def test_experiment_prefix_is_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("DAPM_EXPERIMENT_PREFIX", "pfx")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    prices = synthetic_prices("2024-01-01", "2024-07-01", seed=0)
    run_walk_forward(
        prices, model_params={"max_iter": 5}, n_folds=2,
        test_hours=24 * 7, min_train_hours=24 * 60, experiment="smoke",
    )
    assert len(mlflow.search_runs(experiment_names=["pfx/smoke"])) == 3