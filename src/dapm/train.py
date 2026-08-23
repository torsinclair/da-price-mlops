"""Walk-forward training with MLflow tracking.

Splits are expanding-window and strictly causal: each fold trains on all
data before its test window. Metrics are compared against a lag-24
persistence baseline, the standard sanity check for day-ahead prices.
"""

import argparse
import os

import matplotlib
import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from dapm.config import announce, experiment_name
from dapm.data import load_prices
from dapm.features import make_supervised

matplotlib.use("Agg")

MODEL_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "min_samples_leaf": 20}


def walk_forward_splits(
    index: pd.DatetimeIndex,
    n_folds: int = 6,
    test_hours: int = 24 * 28,
    min_train_hours: int = 24 * 180,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window splits: consecutive test blocks tiling the series end."""
    n = len(index)
    if n - n_folds * test_hours < min_train_hours:
        raise ValueError("not enough history for the requested folds")
    splits = []
    for k in range(n_folds):
        test_start = n - (n_folds - k) * test_hours
        test_end = test_start + test_hours
        splits.append((index[:test_start], index[test_start:test_end]))
    return splits


def fit_fold(
    x: pd.DataFrame, y: pd.Series, train_idx: pd.DatetimeIndex, model_params: dict
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(random_state=0, **model_params)
    model.fit(x.loc[train_idx], y.loc[train_idx])
    return model


def fold_metrics(y_true: pd.Series, y_pred: pd.Series, baseline_pred: pd.Series) -> dict:
    """Point-forecast metrics plus skill against lag-24 persistence."""
    mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    return {
        "mae": mae,
        "rmse": root_mean_squared_error(y_true, y_pred),
        "baseline_mae": baseline_mae,
        "skill_vs_persistence": 1.0 - mae / baseline_mae,
    }


def forecast_figure(y_true: pd.Series, y_pred: pd.Series):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    tail = y_true.index[-336:]
    ax.plot(tail, y_true.loc[tail], label="actual", linewidth=1)
    ax.plot(tail, y_pred.loc[tail], label="predicted", linewidth=1)
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    fig.tight_layout()
    return fig


def importance_figure(model, x_test: pd.DataFrame, y_test: pd.Series):
    import matplotlib.pyplot as plt

    result = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=0)
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(x_test.columns[order], result.importances_mean[order])
    ax.set_xlabel("permutation importance (test fold)")
    fig.tight_layout()
    return fig


def run_walk_forward(
    prices: pd.Series,
    *,
    model_params: dict | None = None,
    n_folds: int = 6,
    test_hours: int = 24 * 28,
    min_train_hours: int = 24 * 180,
    experiment: str = "da-price-forecast",
) -> pd.DataFrame:
    """Train/evaluate across folds, logging everything to MLflow; return fold metrics."""
    model_params = MODEL_PARAMS if model_params is None else model_params
    x, y = make_supervised(prices)
    splits = walk_forward_splits(x.index, n_folds, test_hours, min_train_hours)

    mlflow.set_experiment(experiment_name(experiment))
    rows = []
    with mlflow.start_run(run_name="walk_forward"):
        mlflow.log_params(model_params)
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "test_hours": test_hours,
                "min_train_hours": min_train_hours,
                "data_start": str(prices.index.min()),
                "data_end": str(prices.index.max()),
            }
        )
        for k, (train_idx, test_idx) in enumerate(splits):
            model = fit_fold(x, y, train_idx, model_params)
            pred = pd.Series(model.predict(x.loc[test_idx]), index=test_idx)
            metrics = fold_metrics(y.loc[test_idx], pred, x.loc[test_idx, "lag_24"])
            with mlflow.start_run(run_name=f"fold_{k}", nested=True):
                mlflow.log_params({"fold": k, "train_end": str(train_idx[-1])})
                mlflow.log_metrics(metrics)
            rows.append(metrics)

        summary = pd.DataFrame(rows)
        mlflow.log_metrics({f"mean_{c}": summary[c].mean() for c in summary.columns})
        mlflow.log_figure(forecast_figure(y.loc[test_idx], pred), "final_fold_forecast.png")
        mlflow.log_figure(
            importance_figure(model, x.loc[test_idx], y.loc[test_idx]),
            "permutation_importance.png",
        )
        mlflow.sklearn.log_model(
            model, name="model", input_example=x.loc[test_idx].iloc[:5].astype("float64")
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    announce()
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
    summary = run_walk_forward(prices)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()