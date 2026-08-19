"""Pure feature transforms: calendar, lags, rolling stats.

Causality rule: features for hour t use only prices up to t-24h,
mirroring the day-ahead auction timeline (bids for day D are placed
during D-1). Calendar features are computed in local time
(Europe/Stockholm) because price shape follows local human activity;
the index itself stays UTC per the data contract.
"""

import pandas as pd

LAGS = (24, 48, 168)
ROLLING_WINDOW = 168
MIN_LAG = 24


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in local time, keyed on the UTC index."""
    local = index.tz_convert("Europe/Stockholm")
    return pd.DataFrame(
        {
            "hour": local.hour,
            "dayofweek": local.dayofweek,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "month": local.month,
        },
        index=index,
    )


def lag_features(prices: pd.Series, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Lagged prices; every lag must respect the 24h embargo."""
    if min(lags) < MIN_LAG:
        raise ValueError(f"lags must be >= {MIN_LAG}h to respect the auction embargo")
    return pd.DataFrame({f"lag_{h}": prices.shift(h) for h in lags}, index=prices.index)


def rolling_features(prices: pd.Series, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Trailing rolling stats over the embargoed series."""
    shifted = prices.shift(MIN_LAG)
    r = shifted.rolling(window)
    return pd.DataFrame(
        {f"roll_mean_{window}": r.mean(), f"roll_std_{window}": r.std()},
        index=prices.index,
    )


def make_supervised(prices: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the feature matrix and aligned target, dropping the warmup."""
    x = pd.concat(
        [calendar_features(prices.index), lag_features(prices), rolling_features(prices)],
        axis=1,
    )
    mask = x.notna().all(axis=1)
    return x[mask], prices[mask]