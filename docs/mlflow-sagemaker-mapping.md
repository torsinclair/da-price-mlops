# Mapping the loop onto AWS SageMaker AI

This repo runs its full loop on open-source MLflow (local, sqlite) and
on Databricks managed MLflow (Unity Catalog registry). This note maps
the same loop onto AWS SageMaker AI — studied, not yet run — so the
portability story covers the third major backend honestly.

## Current state (August 2026)

- "SageMaker" classic ML capabilities live under the SageMaker AI name
  (rebranded December 2024).
- SageMaker AI ships fully managed MLflow (GA June 2024). The current
  form is the serverless "MLflow App" — no additional cost, automatic
  scaling and version upgrades (MLflow 3.4 at time of writing),
  available in eu-north-1 (Stockholm) — and AWS recommends it over the
  older instance-based MLflow Tracking Servers.
- Models registered in managed MLflow appear automatically in the
  SageMaker Model Registry, and MLflow-flavored models deploy to
  SageMaker Inference without custom containers.
- Consequence: "MLflow vs SageMaker" is a false choice in 2026. AWS
  itself ships MLflow as the experimentation/registry API, with
  SageMaker as the substrate beneath it. The skills transfer directly.

## The mapping

| This repo (MLflow) | SageMaker AI equivalent | Trading practice |
|---|---|---|
| Experiment tracking (`dapm.train`, nested fold runs) | Managed MLflow App: same MLflow SDK, tracking URI = App ARN, auth via IAM (`sagemaker-mlflow` plugin) | Research journal / backtest log |
| Registry with `@champion`/`@challenger` aliases | Managed MLflow registry, mirrored into SageMaker Model Registry (model package groups; versions carry approval statuses) | Versioned strategy configs with go-live sign-off |
| Margin-gated promotion (`dapm.promote`) | Pipelines condition step gating model registration; or EventBridge rule on registry approval driving deployment | Challenger must beat incumbent to take capital |
| Drift monitoring (`dapm.drift`, `dapm.evaluate`) | Model Monitor: a baseline job freezes reference statistics (cf. our frozen PSI bins), then scheduled monitoring jobs check captured inference data | Daily live-vs-expected reconciliation |
| Scheduled retraining loop (`dapm.loop` + Actions cron) | Pipelines on an EventBridge schedule; Model Monitor alarms via CloudWatch can trigger the retraining pipeline | Systematic re-fit cadence |
| Model artifact (sklearn flavor + signature) | Same artifact; deploys to real-time, serverless, async, or batch endpoints | Execution layer |
| Backend switch via sourced env file | Tracking URI becomes the App ARN; IAM role replaces the OAuth profile | Mandate and permissioning |

## Where the mapping is not one-to-one

1. **Approval semantics.** SageMaker Model Registry speaks
   approval statuses (pending/approved/rejected) — coarser than
   arbitrary aliases. Champion/challenger maps onto it, but the
   vocabulary differs and the gate logic would live in a pipeline
   step rather than registry metadata.
2. **Serving is first-class there, absent here.** This repo stops at
   registry + monitoring; SageMaker's center of gravity is endpoints.
   Honest gap, deliberate scope.
3. **Model Monitor assumes endpoint data capture.** Our drift runs
   offline on batch predictions. Same mathematics (baseline
   constraints ≈ frozen reference distributions), different plumbing.
4. **IAM everywhere.** The deepest practical difference from
   Databricks profiles or local trust — auth design precedes any
   architecture on AWS.
5. **Containers underneath.** Classic SageMaker training/serving
   thinks in ECR images; managed MLflow's value is precisely that
   MLflow-flavored models don't require building them.