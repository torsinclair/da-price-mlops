"""Loop tests: bootstrap + calm steady-state, and a genuine regime change."""

import mlflow
import pandas as pd

from dapm.data import synthetic_prices
from dapm.loop import run_cycle

SPLIT = {"model_params": {"max_iter": 60}, "n_folds": 2, "test_hours": 24 * 7,
         "min_train_hours": 24 * 60}


def test_cycle_bootstraps_then_stays_calm(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    prices = synthetic_prices("2023-01-01", "2024-07-01", seed=0)

    first = run_cycle(prices, experiment="loop-calm", **SPLIT)
    assert first["bootstrapped"] is True
    assert first["drift_retrain"] is False
    assert first["champion_version"] == 1

    second = run_cycle(prices, experiment="loop-calm", **SPLIT)
    assert second["bootstrapped"] is False
    assert second["gate"] is None
    assert second["champion_version"] == 1


def test_cycle_regime_change_retrains_and_promotes(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    prices = synthetic_prices("2023-01-01", "2024-07-01", seed=0)
    run_cycle(prices, experiment="loop-regime", **SPLIT)  # champion of the old regime

    shifted = prices.copy()
    cut = prices.index.max() - pd.Timedelta(days=28)
    shifted[shifted.index > cut] = shifted[shifted.index > cut] + 40.0

    out = run_cycle(shifted, experiment="loop-regime", **SPLIT)
    assert out["drift_retrain"] is True
    assert out["gate"]["promoted"] is True
    assert out["champion_version"] == 2