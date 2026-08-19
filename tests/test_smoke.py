"""Smoke tests: the environment and package skeleton hold together."""

import importlib

MODULES = [
    "dapm.data",
    "dapm.features",
    "dapm.train",
    "dapm.evaluate",
    "dapm.drift",
    "dapm.promote",
]


def test_all_modules_import():
    for name in MODULES:
        importlib.import_module(name)


def test_platform_stack_importable():
    import entsoe  # noqa: F401
    import mlflow  # noqa: F401
    import sklearn  # noqa: F401