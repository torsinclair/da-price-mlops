"""Scheduled monitoring loop: score, and on a drift verdict, retrain and gate.

Orchestration lives here as tested code; the scheduler (GitHub Actions)
stays a dumb invoker. Each cycle: ensure a champion exists (bootstrap on
a fresh registry), score it, and on a retrain verdict train a challenger
and submit it to the promotion gate. A false alarm therefore costs one
refused challenger, never a deployment.
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient

from dapm.data import load_prices
from dapm.evaluate import score_champion
from dapm.promote import champion_version, promote_latest
from dapm.train import run_walk_forward


def run_cycle(
    prices,
    *,
    model_params: dict | None = None,
    n_folds: int = 6,
    test_hours: int = 24 * 28,
    min_train_hours: int = 24 * 180,
    experiment: str = "da-price-forecast",
) -> dict:
    """One monitoring cycle; returns a summary of what happened and why."""
    split = {"n_folds": n_folds, "test_hours": test_hours, "min_train_hours": min_train_hours}
    client = MlflowClient()
    bootstrapped = champion_version(client) is None
    if bootstrapped:
        run_walk_forward(prices, model_params=model_params, experiment=experiment, **split)
        promote_latest(prices, experiment=experiment, **split)

    report = score_champion(prices)
    gate = None
    if report["retrain"]:
        run_walk_forward(prices, model_params=model_params, experiment=experiment, **split)
        gate = promote_latest(prices, experiment=experiment, **split)

    return {
        "bootstrapped": bootstrapped,
        "scored_champion_version": report["champion_version"],
        "drift_retrain": bool(report["retrain"]),
        "gate": gate,
        "champion_version": champion_version(client),
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
    summary = run_cycle(prices)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()