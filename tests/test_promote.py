"""Registry gate tests: pure decision logic plus a hermetic end-to-end flow."""

import mlflow
from mlflow import MlflowClient

from dapm.config import model_name
from dapm.data import synthetic_prices
from dapm.promote import promote_latest, should_promote
from dapm.train import run_walk_forward

SPLIT = {"n_folds": 2, "test_hours": 24 * 7, "min_train_hours": 24 * 60}


def test_gate_requires_margin():
    assert should_promote(10.0, 9.0)  # 10% better: promote
    assert not should_promote(10.0, 9.9)  # 1% better < 2% margin: hold
    assert not should_promote(10.0, 10.0)  # tie goes to the incumbent
    assert not should_promote(10.0, 11.0)


def test_cold_start_then_worse_challenger_rejected(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    prices = synthetic_prices("2024-01-01", "2024-07-01", seed=0)

    run_walk_forward(prices, model_params={"max_iter": 60}, experiment="gate", **SPLIT)
    first = promote_latest(prices, experiment="gate", **SPLIT)
    assert first["promoted"] is True
    assert first["challenger_version"] == 1
    assert first["incumbent_version"] is None

    run_walk_forward(
        prices, model_params={"max_iter": 2, "learning_rate": 0.01}, experiment="gate", **SPLIT
    )
    second = promote_latest(prices, experiment="gate", **SPLIT)
    assert second["promoted"] is False
    assert second["challenger_mae"] > second["champion_mae"]

    client = MlflowClient()
    assert int(client.get_model_version_by_alias(model_name(), "champion").version) == 1
    assert int(client.get_model_version_by_alias(model_name(), "challenger").version) == 2