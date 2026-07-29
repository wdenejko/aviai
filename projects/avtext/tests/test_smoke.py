"""Smoke test: the package installs and every layer is importable.

Trivial on purpose. Its job is to fail loudly if the workspace/package wiring
breaks (bad pyproject, missing __init__, uninstalled package) — not to test
behavior. Real tests arrive with real code, layer by layer.
"""

import importlib

import avtext


def test_version_is_present():
    assert isinstance(avtext.__version__, str)
    assert avtext.__version__


def test_all_layers_import():
    for layer in (
        "ingest",
        "schema",
        "oracles",
        "quality",
        "consensus",
        "tasks",
        "harness",
        "finetune",
    ):
        importlib.import_module(f"avtext.{layer}")
