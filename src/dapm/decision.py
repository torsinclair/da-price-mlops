"""Decision-quality evaluation: does forecast accuracy become realized savings?

A charging task (k hours per local day) is scheduled three ways: against
the forecast, against actuals (perfect foresight, the floor), and naive
charge-on-arrival at 18:00 local (the dumb baseline). Scheduling consumes
only the forecast's within-day ranking of hours, not its level - so MAE
and realized savings can disagree, and when they do, savings win.

Days are grouped in local time to match delivery days; DST transition
days (23/25 local hours) are skipped as a documented simplification.
"""

import argparse
import os

import matplotlib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

from dapm.config import announce, experiment_name, model_name
from dapm.data import load_prices
from dapm.features import make_supervised

matplotlib.use("Agg")

DECISION_EXPERIMENT = "da-price-decision"
HOURS_PER_DAY = 4
NAIVE_START_HOUR = 18
LOCAL_TZ = "Europe/Stockholm"


def charging_schedule(day_prices: pd.Series, hours_needed: int) -> pd.DatetimeIndex:
    """The k cheapest hours of one day's prices, in time order."""
    return day_prices.nsmallest(hours_needed).index.sort_values()


def naive_schedule(day_prices: pd.Series, hours_needed: int, start_hour: int) -> pd.DatetimeIndex:
    """Charge-on-arrival: k consecutive hours from start_hour local."""
    local_hour = day_prices.index.tz_convert(LOCAL_TZ).hour
    mask = (local_hour >= start_hour) & (local_hour < start_hour + hours_needed)
    return day_prices.index[mask]


def daily_costs(
    actual: pd.Series,
    forecast: pd.Series,
    hours_per_day: int = HOURS_PER_DAY,
    naive_start_hour: int = NAIVE_START_HOUR,
) -> pd.DataFrame:
    """Realized cost per local day under forecast, perfect, and naive scheduling.

    All three schedules are priced at actuals; only the selection differs.
    """
    local_day = actual.index.tz_convert(LOCAL_TZ).date
    rows = []
    for day, day_actual in actual.groupby(local_day):
        if len(day_actual) != 24:
            continue  # DST transition or partial edge day
        day_forecast = forecast.loc[day_actual.index]
        rows.append(
            {
                "day": day,
                "cost_forecast": actual.loc[
                    charging_schedule(day_forecast, hours_per_day)
                ].sum(),
                "cost_perfect": actual.loc[
                    charging_schedule(day_actual, hours_per_day)
                ].sum(),
                "cost_naive": actual.loc[
                    naive_schedule(day_actual, hours_per_day, naive_start_hour)
                ].sum(),
            }
        )
    return pd.DataFrame(rows).set_index("day")


def decision_report(costs: pd.DataFrame) -> dict:
    """Aggregate decision-quality metrics over the evaluated days."""
    totals = costs.sum()
    achievable = totals["cost_naive"] - totals["cost_perfect"]
    captured = totals["cost_naive"] - totals["cost_forecast"]
    return {
        "n_days": len(costs),
        "cost_forecast": float(totals["cost_forecast"]),
        "cost_perfect": float(totals["cost_perfect"]),
        "cost_naive": float(totals["cost_naive"]),
        "cost_regret": float(totals["cost_forecast"] - totals["cost_perfect"]),
        "savings_capture": float(captured / achievable) if achievable > 0 else float("nan"),
    }


def cumulative_cost_figure(costs: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for column, label in [
        ("cost_naive", "naive 18:00"),
        ("cost_forecast", "forecast schedule"),
        ("cost_perfect", "perfect foresight"),
    ]:
        ax.plot(costs.index, costs[column].cumsum(), label=label, linewidth=1)
    ax.set_ylabel("cumulative cost (EUR/MWh-hours)")
    ax.legend()
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    announce()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "entsoe"], default="synthetic")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--days", type=int, default=90, help="trailing days to evaluate")
    args = parser.parse_args(argv)

    api_key = None
    if args.source == "entsoe":
        from dotenv import load_dotenv

        load_dotenv(".env")
        api_key = os.environ["ENTSOE_API_KEY"]

    prices = load_prices(args.start, args.end, source=args.source, api_key=api_key)
    x, y = make_supervised(prices)
    model = mlflow.sklearn.load_model(f"models:/{model_name()}@champion")

    window = x.index >= x.index.max() - pd.Timedelta(days=args.days)
    forecast = pd.Series(model.predict(x[window]), index=x.index[window])
    actual = y[window]

    costs = daily_costs(actual, forecast)
    report = decision_report(costs)
    report["forecast_mae"] = float(mean_absolute_error(actual, forecast))

    mlflow.set_experiment(experiment_name(DECISION_EXPERIMENT))
    with mlflow.start_run(run_name="decision"):
        mlflow.log_params(
            {"hours_per_day": HOURS_PER_DAY, "naive_start_hour": NAIVE_START_HOUR,
             "days": args.days}
        )
        mlflow.log_metrics(report)
        mlflow.log_figure(cumulative_cost_figure(costs), "cumulative_cost.png")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()