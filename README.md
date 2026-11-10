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
5. **Decision quality** (`dapm.decision`): the champion's forecasts
   drive a daily charging schedule (k cheapest hours), evaluated at
   realized prices against perfect foresight and naive
   charge-on-arrival. The product metric is savings capture: the
   fraction of the naive-to-perfect gap the forecast collects.

## Quickstart

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q                      # 39 tests, hermetic (tmp sqlite per test)
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
  a test.
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
- **Ephemeral backend per scheduled run.** The Actions loop runs
  hermetically against a fresh sqlite backend; pointing the same job at
  a persistent tracking server is a one-env-var change
  (`MLFLOW_TRACKING_URI`), which is the intended production shape.
- **That production shape is declared, not just described.** `infra/` is
  the preceding sentence as Terraform: Postgres Flexible Server as the
  backend store, a private blob container as the artifact store, Key Vault
  for the DSN and the ENTSO-E token, a user-assigned identity because the
  storage account's shared keys are disabled, and the tracking server on
  Container Apps. The outputs are the same environment variables the
  application already reads, so the infrastructure and the config speak one
  vocabulary. CI runs `fmt -check`, `init -backend=false` and `validate` on
  every change under `infra/`. It has never been applied against a live
  subscription, and `infra/README.md` leads with that rather than letting
  the directory imply otherwise.
- **Model metrics and product metrics are different experiments.** MAE
  lives in training/scoring; savings capture and cost regret live in
  the decision experiment. A uniformly biased forecast has terrible MAE
  and schedules perfectly — the unit test proving it is the reason this
  layer exists. On synthetic data the champion captures ~90% of
  achievable savings at 22% MAE skill: decision value concentrates in
  the within-day ranking, not the level.
- **Portable across backends, honestly accounted.** The same entry
  points run against local sqlite and Databricks (hosted MLflow +
  Unity Catalog registry). The marketing version is "change the
  tracking URI"; the lived version is five environment variables
  (tracking URI, registry URI, auth profile, experiment path prefix,
  three-level UC model name — see the `.env.databricks` pattern),
  plus two requirements UC enforces in code: registered models must
  carry a signature, and model names are SQL identifiers. Because
  env-driven config fails silent, every entry point announces its
  resolved target before running. A mapping of the same loop onto
  AWS SageMaker AI is in `docs/mlflow-sagemaker-mapping.md`.
    
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