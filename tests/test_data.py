"""Tests for the synthetic price generator against the data contract."""

import numpy as np
import pandas as pd

from dapm.data import synthetic_prices


def test_honors_data_contract():
    s = synthetic_prices("2024-01-01", "2024-02-01", seed=1)
    assert len(s) == 31 * 24
    assert s.index.tz is not None and str(s.index.tz) == "UTC"
    assert s.index.is_monotonic_increasing
    assert s.name == "price_eur_mwh"
    assert pd.infer_freq(s.index) == "h"


def test_reproducible_by_seed():
    a = synthetic_prices("2024-01-01", "2024-03-01", seed=7)
    b = synthetic_prices("2024-01-01", "2024-03-01", seed=7)
    c = synthetic_prices("2024-01-01", "2024-03-01", seed=8)
    assert a.equals(b)
    assert not a.equals(c)


def test_values_finite_and_plausible():
    s = synthetic_prices("2024-01-01", "2025-01-01", seed=3)
    assert np.isfinite(s).all()
    assert 30 < s.mean() < 70


def test_weekend_seasonality_exists():
    s = synthetic_prices("2024-01-01", "2024-07-01", seed=5)
    weekday = s[s.index.dayofweek < 5].mean()
    weekend = s[s.index.dayofweek >= 5].mean()
    assert weekday > weekend