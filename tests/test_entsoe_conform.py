"""Tests for ENTSO-E conformance logic (no network required)."""

import pandas as pd

from dapm.data import _conform


def test_conform_local_tz_to_utc_hourly():
    idx = pd.date_range("2024-06-01", periods=48, freq="h", tz="Europe/Stockholm")
    raw = pd.Series(range(48), index=idx, dtype=float)
    s = _conform(raw)
    assert str(s.index.tz) == "UTC"
    assert pd.infer_freq(s.index) == "h"
    assert s.name == "price_eur_mwh"


def test_conform_15min_resampled_to_hourly_means():
    idx = pd.date_range("2025-11-01", periods=8, freq="15min", tz="Europe/Stockholm")
    raw = pd.Series([10, 20, 30, 40, 50, 50, 50, 50], index=idx, dtype=float)
    s = _conform(raw)
    assert len(s) == 2
    assert s.iloc[0] == 25.0
    assert s.iloc[1] == 50.0


def test_conform_handles_dst_spring_forward():
    # 2024-03-31: Europe/Stockholm skips 02:00; the local day has 23 hours.
    idx = pd.date_range("2024-03-30 12:00", "2024-04-01 12:00", freq="h", tz="Europe/Stockholm")
    raw = pd.Series(1.0, index=idx)
    s = _conform(raw)
    assert str(s.index.tz) == "UTC"
    assert s.index.is_unique and s.index.is_monotonic_increasing


def test_conform_trims_end_inclusive_upstream_to_half_open():
    # Upstream returns the end boundary as an extra row; contract excludes it.
    idx = pd.date_range("2024-01-01", "2024-01-02", periods=25, freq=None, tz="UTC")
    raw = pd.Series(1.0, index=idx)
    s = _conform(raw, end=pd.Timestamp("2024-01-02", tz="UTC"))
    assert len(s) == 24
    assert s.index.max() == pd.Timestamp("2024-01-01 23:00", tz="UTC")