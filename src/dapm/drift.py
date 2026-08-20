"""Distribution and performance drift checks.

Three detectors with three latencies: PSI on features (input shift,
available immediately), KS on the target (outcome shift, available as
prices realize), and a residual monitor (performance shift, available
only once predictions meet ground truth). PSI bins are frozen from the
reference window so the yardstick cannot drift with the data.
PSI comparison windows must be calendar-matched: deterministic calendar
features (month, hour) shift by construction when windows cover
different parts of the year, and that is a false alarm, not drift.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN = 0.10
PSI_ALERT = 0.25
KS_PVALUE_ALERT = 0.01
RESIDUAL_DEGRADATION_ALERT = 0.25
N_BINS = 10
_EPS = 1e-6


def psi(reference: pd.Series, live: pd.Series, n_bins: int = N_BINS) -> float:
    """Population Stability Index of live vs reference, on frozen bins.

    Bin edges are the reference distribution's quantiles; outermost edges
    are widened to infinity so live values outside the reference range
    land in the edge bins instead of disappearing.
    """
    edges = np.quantile(reference.dropna(), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference.dropna(), bins=edges)[0] / max(len(reference.dropna()), 1)
    live_frac = np.histogram(live.dropna(), bins=edges)[0] / max(len(live.dropna()), 1)
    ref_frac = np.clip(ref_frac, _EPS, None)
    live_frac = np.clip(live_frac, _EPS, None)
    return float(np.sum((live_frac - ref_frac) * np.log(live_frac / ref_frac)))


def feature_psi(reference: pd.DataFrame, live: pd.DataFrame) -> pd.Series:
    """PSI per feature column, live vs reference."""
    return pd.Series({c: psi(reference[c], live[c]) for c in reference.columns}, name="psi")


def target_ks(reference: pd.Series, live: pd.Series) -> dict:
    """Two-sample KS test of live target vs reference target."""
    stat, pvalue = ks_2samp(reference.dropna(), live.dropna())
    return {"ks_stat": float(stat), "ks_pvalue": float(pvalue)}


def residual_degradation(live_mae: float, expected_mae: float) -> float:
    """Relative MAE degradation of live performance vs the champion's holdout MAE."""
    return live_mae / expected_mae - 1.0


def drift_report(
    reference_x: pd.DataFrame,
    live_x: pd.DataFrame,
    reference_y: pd.Series,
    live_y: pd.Series,
    live_mae: float,
    expected_mae: float,
) -> dict:
    """All three detectors plus a single retrain verdict.

    Retrain triggers on any alert: a feature PSI above PSI_ALERT, a target
    KS p-value below KS_PVALUE_ALERT, or live MAE degraded more than
    RESIDUAL_DEGRADATION_ALERT relative to the champion's holdout MAE.
    """
    per_feature = feature_psi(reference_x, live_x)
    ks = target_ks(reference_y, live_y)
    degradation = residual_degradation(live_mae, expected_mae)
    return {
        "max_feature_psi": float(per_feature.max()),
        "worst_feature": str(per_feature.idxmax()),
        "n_features_warn": int((per_feature > PSI_WARN).sum()),
        **ks,
        "residual_degradation": degradation,
        "retrain": bool(
            per_feature.max() > PSI_ALERT
            or ks["ks_pvalue"] < KS_PVALUE_ALERT
            or degradation > RESIDUAL_DEGRADATION_ALERT
        ),
    }