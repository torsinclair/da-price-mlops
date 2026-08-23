"""Deployment-target configuration: names resolve from the environment.

Local MLflow accepts flat names; Databricks requires workspace-path
experiment names (/Users/<you>/...) and Unity Catalog three-level model
names (catalog.schema.model). Resolving names per call keeps identical
code runnable against both targets:

    DAPM_MODEL_NAME          overrides the registered model name
    DAPM_EXPERIMENT_PREFIX   is prepended to every experiment name
"""

import os


def model_name() -> str:
    return os.environ.get("DAPM_MODEL_NAME", "da-price-forecaster")


def experiment_name(default: str) -> str:
    prefix = os.environ.get("DAPM_EXPERIMENT_PREFIX", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix + default


def announce() -> None:
    """Print the resolved deployment target; silent fallback is how runs land
    on the wrong backend unnoticed."""
    import mlflow

    print(
        f"[dapm] tracking={mlflow.get_tracking_uri()} "
        f"registry={mlflow.get_registry_uri()} model={model_name()!r} "
        f"prefix={os.environ.get('DAPM_EXPERIMENT_PREFIX', '')!r}"
    )