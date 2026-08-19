"""Load day-ahead price series: ENTSO-E SE3 or synthetic fallback.

Data contract (both paths): hourly pd.Series of EUR/MWh, tz-aware UTC
index, half-open interval [start, end), named "price_eur_mwh".
"""

import numpy as np
import pandas as pd


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