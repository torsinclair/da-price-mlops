"""Champion/challenger promotion gate against the registry.

Deployment state lives in registry aliases (@champion, @challenger), not
the deprecated stage machinery: versions are immutable, aliases are
mutable pointers, and moving @champion is the single auditable act of
deployment. A challenger earns promotion only by beating the incumbent
on the same, most recent holdout window by a required margin; ties go
to the incumbent.
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error

from dapm.config import model_name
from dapm.data import load_prices
from dapm.features import make_supervised
from dapm.train import walk_forward_splits

MIN_RELATIVE_IMPROVEMENT = 0.02


def should_promote(
    champion_mae: float,
    challenger_mae: float,
    min_relative_improvement: float = MIN_RELATIVE_IMPROVEMENT,
) -> bool:
    """Strict gate: the challenger must beat the incumbent by the margin."""
    return challenger_mae < champion_mae * (1.0 - min_relative_improvement)


def latest_run_id(experiment: str) -> str:
    """Id of the most recent walk_forward parent run in the experiment."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="attributes.run_name = 'walk_forward'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"no walk_forward runs found in experiment {experiment!r}")
    return runs.iloc[0]["run_id"]


def latest_model_id(experiment: str, run_id: str) -> str:
    """Model id of the LoggedModel produced by the given run."""
    exp = mlflow.get_experiment_by_name(experiment)
    models = mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id], output_format="list"
    )
    ids = [m.model_id for m in models if m.source_run_id == run_id]
    if not ids:
        raise ValueError(f"no logged models found for run {run_id}")
    return ids[0]


def champion_version(client: MlflowClient) -> int | None:
    try:
        return int(client.get_model_version_by_alias(model_name(), "champion").version)
    except MlflowException:
        return None


def promote_latest(
    prices,
    *,
    n_folds: int = 6,
    test_hours: int = 24 * 28,
    min_train_hours: int = 24 * 180,
    experiment: str = "da-price-forecast",
) -> dict:
    """Register the latest run's model, gate it against @champion, move aliases.

    Both models are evaluated on the most recent test window, which is
    out-of-sample for both by construction.
    """
    client = MlflowClient()
    x, y = make_supervised(prices)
    _, test_idx = walk_forward_splits(x.index, n_folds, test_hours, min_train_hours)[-1]
    x_test, y_test = x.loc[test_idx], y.loc[test_idx]

    run_id = latest_run_id(experiment)
    model_id = latest_model_id(experiment, run_id)
    version = int(mlflow.register_model(f"models:/{model_id}", model_name()).version)
    client.set_registered_model_alias(model_name(), "challenger", version)

    challenger = mlflow.sklearn.load_model(f"models:/{model_name()}@challenger")
    challenger_mae = float(mean_absolute_error(y_test, challenger.predict(x_test)))

    incumbent = champion_version(client)
    if incumbent is None:
        champion_mae = None
        promoted = True  # cold start: an evaluated model beats no model
    else:
        champion = mlflow.sklearn.load_model(f"models:/{model_name()}@champion")
        champion_mae = float(mean_absolute_error(y_test, champion.predict(x_test)))
        promoted = should_promote(champion_mae, challenger_mae)

    if promoted:
        client.set_registered_model_alias(model_name(), "champion", version)

    tags = {"gate_challenger_mae": f"{challenger_mae:.4f}", "gate_promoted": str(promoted)}
    if incumbent is not None:
        tags["gate_champion_version"] = str(incumbent)
        tags["gate_champion_mae"] = f"{champion_mae:.4f}"
    for key, value in tags.items():
        client.set_model_version_tag(model_name(), str(version), key, value)

    return {
        "challenger_version": version,
        "incumbent_version": incumbent,
        "champion_mae": champion_mae,
        "challenger_mae": challenger_mae,
        "promoted": promoted,
    }


def main(argv: list[str] | None = None) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "entsoe"], default="synthetic")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args(argv)

    api_key = None
    if args.source == "entsoe":
        from dotenv import load_dotenv

        load_dotenv(".env")
        api_key = os.environ["ENTSOE_API_KEY"]

    prices = load_prices(args.start, args.end, source=args.source, api_key=api_key)
    report = promote_latest(prices)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()