"""Decision-layer tests: the level-vs-ranking thesis, and control schedules."""

import numpy as np
import pandas as pd

from dapm.data import synthetic_prices
from dapm.decision import charging_schedule, daily_costs, decision_report, naive_schedule


def _day(seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-06-03", periods=24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(20, 80, 24), index=idx)


def test_schedule_picks_cheapest_hours():
    day = _day()
    picked = charging_schedule(day, 4)
    assert len(picked) == 4
    assert set(day.loc[picked]) == set(day.nsmallest(4))


def test_uniformly_biased_forecast_schedules_perfectly():
    """+20 EUR everywhere: MAE is awful, the schedule is identical to perfect."""
    prices = synthetic_prices("2024-05-01", "2024-06-01", seed=2)
    costs = daily_costs(prices, prices + 20.0)
    report = decision_report(costs)
    assert report["cost_regret"] == 0.0
    assert report["savings_capture"] == 1.0


def test_adversarial_forecast_captures_nothing():
    """Inverted ranking: schedules the *most* expensive hours."""
    prices = synthetic_prices("2024-05-01", "2024-06-01", seed=2)
    costs = daily_costs(prices, -prices)
    report = decision_report(costs)
    assert report["cost_regret"] > 0
    assert report["savings_capture"] < 0


def test_perfect_is_floor_and_naive_evening_is_expensive():
    prices = synthetic_prices("2024-05-01", "2024-08-01", seed=5)
    costs = daily_costs(prices, prices)
    assert (costs["cost_perfect"] <= costs["cost_forecast"] + 1e-9).all()
    assert (costs["cost_perfect"] <= costs["cost_naive"] + 1e-9).all()
    report = decision_report(costs)
    assert report["cost_naive"] > report["cost_perfect"]


def test_naive_schedule_is_evening_hours():
    day = _day()
    picked = naive_schedule(day, 4, 18)
    local_hours = picked.tz_convert("Europe/Stockholm").hour
    assert list(local_hours) == [18, 19, 20, 21]