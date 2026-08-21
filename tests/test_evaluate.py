"""Scoring entry point tests: window math plus a hermetic end-to-end score."""

import mlflow
import pandas as pd

from dapm.data import synthetic_prices
from dapm.evaluate import reference_window, score_champion
from dapm.promote import promote_latest
from dapm.train import run_walk_forward

SPLIT = {"n_folds": 2, "test_hours": 24 * 7, "min_train_hours": 24 * 60}


def test_reference_window_is_calendar_matched():
    start, end = reference_window(
        pd.Timestamp("2024-06-03", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
    )
    assert start == pd.Timestamp("2023-06-03", tz="UTC")
    assert end == pd.Timestamp("2023-07-01", tz="UTC")


def test_score_champion_end_to_end(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    prices = synthetic_prices("2023-01-01", "2024-07-01", seed=0)
    run_walk_forward(prices, model_params={"max_iter": 60}, experiment="score-e2e", **SPLIT)
    promote_latest(prices, experiment="score-e2e", **SPLIT)

    report = score_champion(prices)
    assert report["champion_version"] == 1
    assert report["retrain"] is False  # calm: calendar-matched, slow features declared
    assert report["max_level_shift"] < 1.0

    runs = mlflow.search_runs(experiment_names=["da-price-scoring"])
    assert len(runs) == 1