# da-price-mlops

Day-ahead electricity price forecasting for the SE3 (Stockholm) bidding
zone, built as a complete ML-platform loop: experiment tracking, model
registry, gated promotion, drift monitoring, and (in progress) automated
retraining. The model is deliberately simple — the platform machinery is
the exhibit.

The validation and governance
patterns come from systematic trading practice; this repo maps them
onto standard MLOps tooling.

## The loop

1. **Train** (`dapm.train`): expanding-window walk-forward folds, each
   logged to MLflow as a nested run; metrics include skill against a
   lag-24 persistence baseline, the standard sanity check for day-ahead
   prices.
2. **Promote** (`dapm.promote`): the latest model is registered and set
   as `@challenger`; it takes `@champion` only by beating the incumbent
   on the same holdout window by a required margin. Ties go to the
   incumbent. Every verdict is written back as version tags, so the
   registry carries its own audit trail.
3. **Score** (`dapm.evaluate`): the champion scores the trailing live
   window; three drift detectors compare it against a calendar-matched
   reference (same span, previous year) and the report is logged to
   MLflow. The expected-performance bar comes from the champion's own
   registration tags.
4. **Retrain trigger**: a scheduled workflow runs scoring and, on a
   drift verdict, retrains and submits the result to the same promotion
   gate. A false alarm therefore costs one refused challenger, not a
   deployment.

## Quickstart

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q                      # 31 tests, hermetic (tmp sqlite per test)
    python -m dapm.train           # synthetic data by default
    python -m dapm.promote
    python -m dapm.evaluate
    mlflow ui --backend-store-uri sqlite:///mlflow.db

For real data: register at transparency.entsoe.eu, put the token in
`.env` (see `.env.example`), and pass `--source entsoe`.

## Design decisions

- **Walk-forward, never k-fold.** Splits are expanding-window and
  strictly causal. A unit test rewrites all prices within 24h of a
  timestamp and asserts the features do not move — leakage detection as
  a test, not a hope.
- **A 24-hour feature embargo, enforced.** Day-ahead bids for day D are
  placed during D−1, so no feature may use prices newer than t−24h. The
  embargo raises `ValueError` rather than relying on convention.
- **UTC index, local-time calendar features.** ENTSO-E data arrives in
  local time with DST artifacts and, since autumn 2025, 15-minute MTUs;
  ingest conforms everything to an hourly UTC contract. Calendar
  features are computed in Europe/Stockholm because price shape follows
  local human activity.
- **Registry aliases, not stages.** Deployment state is the mutable
  `@champion`/`@challenger` pointers over immutable versions; the stage
  API is deprecated. Moving `@champion` is the single auditable act of
  deployment.
- **SQLite tracking backend.** MLflow 3 has moved the filesystem store
  to maintenance mode, and the registry never worked on it. SQLite is
  the zero-infrastructure database backend; production would swap the
  URI for Postgres.
- **Strict warnings with documented exceptions.** `filterwarnings =
  ["error"]`, with three message-targeted ignores, each carrying its
  cause and expiry condition in a comment. Running Python 3.14 ahead of
  the ecosystem makes this policy earn its keep.
- **Hand-rolled drift detectors.** PSI (frozen reference bins), KS, and
  a residual monitor, ~100 lines total, chosen over a library so every
  threshold is explainable. Evidently was the evaluated alternative.
- **Calendar-matched drift windows.** Deterministic calendar features
  shift by construction when windows cover different parts of the year;
  the first "calm" test scenario false-alarmed on `month` (PSI 5.8)
  until the windows were matched.
- **Slow features get level checks, not PSI.** A 168h rolling feature
  has ~4 independent observations in a 28-day window; PSI on it is
  noise. Caller-declared slow features are monitored by mean shift in
  reference-std units instead.
- **KS alerts on effect size, gated by significance.** At production
  sample sizes p-values flag practically irrelevant shifts; the alert
  requires both p < 0.01 and statistic > 0.15.
- **Sensitive triggers, strict gates.** A false drift alarm costs one
  retrain that the promotion gate then refuses; a missed drift costs
  sustained bad predictions. The asymmetry justifies erring toward
  firing.
- **Functions over classes.** A class must earn its existence; none has
  yet.

## Known simplifications

- The parquet cache key does not include a code version; a change to
  ingest logic requires manually clearing `data/cache` (found the hard
  way: an end-inclusive upstream series cached one future hour until
  the contract test was extended).
- In the synthetic demo the scoring window partially overlaps training
  data, flattering the residual monitor; with live data the window
  postdates training by definition.
- The level-shift denominator uses the reference window's own std,
  which is itself noisy on short windows; scaling it from a longer
  history is the known next improvement.

## Data

Both sources honor one contract: hourly `pd.Series` of EUR/MWh, tz-aware
UTC, half-open `[start, end)`, named `price_eur_mwh`. The synthetic
generator (seasonality + AR(1) noise + rare spikes) exists so the
pipeline runs without network access; it does not pretend to be SE3,
which has negative hours and fatter tails.