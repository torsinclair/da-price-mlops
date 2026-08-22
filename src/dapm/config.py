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
    return os.environ.get("DAPM_EXPERIMENT_PREFIX", "") + default