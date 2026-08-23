"""Shared fixtures: isolate the suite from ambient deployment configuration."""

import pytest


@pytest.fixture(autouse=True)
def _clean_dapm_env(monkeypatch):
    """Tests must not inherit deployment env from the invoking shell."""
    monkeypatch.delenv("DAPM_EXPERIMENT_PREFIX", raising=False)
    monkeypatch.delenv("DAPM_MODEL_NAME", raising=False)