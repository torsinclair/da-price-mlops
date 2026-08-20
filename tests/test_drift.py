"""Drift detector tests: positive and negative controls for each detector."""

import numpy as np
import pandas as pd

from dapm.data import synthetic_prices
from dapm.drift import drift_report, feature_psi, psi, residual_degradation, target_ks
from dapm.features import make_supervised

RNG = np.random.default_rng(0)


def _series(values):
    return pd.Series(values)


def test_psi_quiet_on_same_distribution():
    a = _series(RNG.normal(0, 1, 4000))
    b = _series(RNG.normal(0, 1, 4000))
    assert psi(a, b) < 0.05


def test_psi_fires_on_shift_and_scale():
    a = _series(RNG.normal(0, 1, 4000))
    shifted = _series(RNG.normal(1.0, 1, 4000))
    widened = _series(RNG.normal(0, 2.0, 4000))
    assert psi(a, shifted) > 0.25
    assert psi(a, widened) > 0.25


def test_psi_handles_live_values_outside_reference_range():
    a = _series(RNG.normal(0, 1, 2000))
    b = _series(RNG.normal(6, 1, 2000))  # entirely outside reference support
    assert np.isfinite(psi(a, b))
    assert psi(a, b) > 1.0


def test_target_ks_controls():
    a = _series(RNG.normal(0, 1, 3000))
    same = _series(RNG.normal(0, 1, 3000))
    other = _series(RNG.normal(0.5, 1, 3000))
    assert target_ks(a, same)["ks_pvalue"] > 0.05
    assert target_ks(a, other)["ks_pvalue"] < 0.01


def test_residual_degradation_sign_and_scale():
    assert residual_degradation(6.0, 6.0) == 0.0
    assert residual_degradation(7.5, 6.0) == 0.25
    assert residual_degradation(5.4, 6.0) < 0.0  # improvement is negative


def test_drift_report_end_to_end_on_regime_change():
    # Calendar-matched windows (Jan-Jun both years): deterministic calendar
    # features like `month` shift by construction when windows cover
    # different parts of the year, which is a false alarm, not drift.
    prices = synthetic_prices("2023-01-01", "2024-07-01", seed=3)
    x, y = make_supervised(prices)
    ref_end = pd.Timestamp("2023-07-01", tz="UTC")
    live_start = pd.Timestamp("2024-01-01", tz="UTC")
    ref_x, live_x = x[x.index < ref_end], x[x.index >= live_start]
    ref_y, live_y = y[y.index < ref_end], y[y.index >= live_start]
    calm = drift_report(ref_x, live_x, ref_y, live_y, live_mae=6.0, expected_mae=6.0)
    assert calm["retrain"] is False

    shocked = drift_report(ref_x, live_x, ref_y, live_y + 40.0, live_mae=9.0, expected_mae=6.0)
    assert shocked["retrain"] is True
    assert shocked["ks_pvalue"] < 0.01
    assert shocked["residual_degradation"] == 0.5


def test_feature_psi_labels_the_drifted_column():
    n = 3000
    ref = pd.DataFrame({"stable": RNG.normal(0, 1, n), "drifted": RNG.normal(0, 1, n)})
    live = pd.DataFrame({"stable": RNG.normal(0, 1, n), "drifted": RNG.normal(2.0, 1, n)})
    result = feature_psi(ref, live)
    assert result["drifted"] > 0.25
    assert result["stable"] < 0.05
    assert result.idxmax() == "drifted"