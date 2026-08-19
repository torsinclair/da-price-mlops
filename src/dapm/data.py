"""Load day-ahead price series: ENTSO-E SE3 or synthetic fallback.

Data contract (both paths): hourly pd.Series of EUR/MWh, tz-aware UTC
index, half-open interval [start, end), named "price_eur_mwh".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient


def synthetic_prices(start: str, end: str, seed: int = 0) -> pd.Series:
    """Generate synthetic hourly day-ahead prices honoring the data contract.

    Structure: base level + two-peak daily shape + weekend damping
    + AR(1) noise + rare positive spikes. Deliberately simple: it exists
    so the pipeline runs end to end without network access, not to fool
    anyone about being real SE3 data (which has negative hours, wind-driven
    regimes, and fatter tails).
    """
    index = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    n = len(index)
    rng = np.random.default_rng(seed)

    hour = index.hour.to_numpy()
    base = 45.0
    daily = 8.0 * np.sin((hour - 6) * 2 * np.pi / 24) + 5.0 * np.sin(
        (hour - 5) * 4 * np.pi / 24
    )
    weekend = np.where(index.dayofweek.to_numpy() >= 5, -6.0, 0.0)

    shocks = rng.normal(0.0, 3.0, n)
    noise = np.empty(n)
    prev = 0.0
    for i in range(n):
        prev = 0.9 * prev + shocks[i]
        noise[i] = prev

    spikes = np.where(rng.random(n) < 0.002, rng.exponential(60.0, n), 0.0)

    return pd.Series(base + daily + weekend + noise + spikes, index=index, name="price_eur_mwh")

def _conform(raw: pd.Series, end: pd.Timestamp | None = None) -> pd.Series:
    """Conform a raw ENTSO-E series to the data contract.

    Handles: local-time index -> UTC; the autumn-2025 switch of day-ahead
    auctions to 15-minute MTUs -> hourly means; unsorted or duplicate
    timestamps around DST transitions -> sorted, unique hourly index;
    end-inclusive upstream series -> half-open [start, end) when end given.
    """
    s = raw.copy()
    s.index = s.index.tz_convert("UTC")
    s = s.sort_index()
    s = s.resample("h").mean().dropna()
    if end is not None:
        s = s[s.index < end]
    s.name = "price_eur_mwh"
    return s

def fetch_entsoe_prices(start: str, end: str, api_key: str, area: str = "SE_3") -> pd.Series:
    """Fetch day-ahead prices from the ENTSO-E Transparency Platform.

    The upstream client returns an end-inclusive series; _conform trims it
    to the half-open [start, end) interval required by the data contract.
    """
    client = EntsoePandasClient(api_key=api_key)
    end_ts = pd.Timestamp(end, tz="UTC")
    raw = client.query_day_ahead_prices(
        area,
        start=pd.Timestamp(start, tz="UTC"),
        end=end_ts,
    )
    return _conform(raw, end=end_ts)


DEFAULT_CACHE_DIR = Path("data/cache")


def load_prices(
    start: str,
    end: str,
    source: str = "synthetic",
    api_key: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.Series:
    """Load prices from the requested source, caching ENTSO-E pulls to parquet."""
    if source == "synthetic":
        return synthetic_prices(start, end)
    if source != "entsoe":
        raise ValueError(f"unknown source: {source!r}")
    if api_key is None:
        raise ValueError("entsoe source requires api_key")

    cache_path = cache_dir / f"entsoe_se3_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)["price_eur_mwh"]

    prices = fetch_entsoe_prices(start, end, api_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_frame().to_parquet(cache_path)
    return prices