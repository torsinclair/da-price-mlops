"""Score the champion on the live window and check for drift.

The scoring job is the monitoring heartbeat: load @champion from the
registry, score the trailing live window, compare against a calendar-
matched reference (same span, previous year) so deterministic calendar
features cannot false-alarm, and log the full drift report to MLflow.
The expected MAE comes from the champion's own gate tag: the registry
carries the performance bar each version was hired at. The retrain
verdict is printed, not encoded in the exit code: drift is a finding,
not a failure.
"""

import argparse
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from dapm.data import load_prices
from dapm.drift import drift_report
from dapm.features import make_supervised
from dapm.promote import MODEL_NAME

SCORING_EXPERIMENT = "da-price-scoring"
LIVE_DAYS = 28
SLOW_FEATURES = ("roll_mean_168", "roll_std_168")


def reference_window(
    live_start: pd.Timestamp, live_end: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar-matched reference: the same span one year earlier."""
    return live_start - pd.DateOffset(years=1), live_end - pd.DateOffset(years=1)


def score_champion(
    prices: pd.Series, *, live_days: int = LIVE_DAYS, experiment: str = SCORING_EXPERIMENT
) -> dict:
    """Score @champion on the trailing window; log the drift report to MLflow."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    expected_mae = float(mv.tags["gate_challenger_mae"])
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    x, y = make_supervised(prices)
    live_end = x.index.max()
    live_start = live_end - pd.Timedelta(days=live_days)
    ref_start, ref_end = reference_window(live_start, live_end)
    if ref_start < x.index.min():
        raise ValueError("series too short for a calendar-matched reference window")

    live = x.index >= live_start
    ref = (x.index >= ref_start) & (x.index < ref_end)
    live_mae = float(mean_absolute_error(y[live], model.predict(x[live])))

    report = drift_report(
        x[ref], x[live], y[ref], y[live], live_mae, expected_mae, slow_features=SLOW_FEATURES
    )

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="score"):
        mlflow.log_params(
            {
                "champion_version": mv.version,
                "live_start": str(live_start),
                "live_end": str(live_end),
                "ref_start": str(ref_start),
                "ref_end": str(ref_end),
            }
        )
        mlflow.set_tag("worst_feature", report["worst_feature"])
        mlflow.set_tag("worst_slow_feature", report["worst_slow_feature"])
        numeric = {
            k: float(v)
            for k, v in report.items()
            if k not in ("worst_feature", "worst_slow_feature")
        }
        mlflow.log_metrics({**numeric, "live_mae": live_mae, "expected_mae": expected_mae})

    return {**report, "live_mae": live_mae, "expected_mae": expected_mae,
            "champion_version": int(mv.version)}


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
    report = score_champion(prices)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()