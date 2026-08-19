"""Tests for feature transforms: shapes, local-time calendar, causality."""

import pandas as pd
import pytest

from dapm.data import synthetic_prices
from dapm.features import MIN_LAG, ROLLING_WINDOW, lag_features, make_supervised


def test_shapes_aligned_and_warmup_dropped():
    s = synthetic_prices("2024-01-01", "2024-03-01", seed=1)
    x, y = make_supervised(s)
    assert len(x) == len(y) == len(s) - (ROLLING_WINDOW + MIN_LAG - 1)
    assert x.notna().all().all()
    assert (x.index == y.index).all()


def test_calendar_hour_is_stockholm_local():
    s = synthetic_prices("2024-01-01", "2024-08-01", seed=1)
    x, _ = make_supervised(s)
    # CET (winter, UTC+1): 10:00 UTC is 11 local. CEST (summer, UTC+2): 12 local.
    assert x.loc[pd.Timestamp("2024-02-15 10:00", tz="UTC"), "hour"] == 11
    assert x.loc[pd.Timestamp("2024-06-15 10:00", tz="UTC"), "hour"] == 12


def test_lag_values_are_exact():
    s = synthetic_prices("2024-01-01", "2024-03-01", seed=4)
    x, _ = make_supervised(s)
    t = pd.Timestamp("2024-02-10 06:00", tz="UTC")
    assert x.loc[t, "lag_24"] == s.loc[t - pd.Timedelta(hours=24)]
    assert x.loc[t, "lag_168"] == s.loc[t - pd.Timedelta(hours=168)]


def test_embargo_violating_lags_rejected():
    s = synthetic_prices("2024-01-01", "2024-02-01", seed=1)
    with pytest.raises(ValueError):
        lag_features(s, lags=(1, 24))


def test_features_are_causal_under_future_perturbation():
    """Rewriting all prices within 24h of t and later must not move features at t."""
    s = synthetic_prices("2024-01-01", "2024-04-01", seed=2)
    x, _ = make_supervised(s)
    t = pd.Timestamp("2024-03-01 12:00", tz="UTC")
    s2 = s.copy()
    s2[s2.index > t - pd.Timedelta(hours=24)] = 9999.0
    x2, _ = make_supervised(s2)
    pd.testing.assert_frame_equal(x.loc[:t], x2.loc[:t])