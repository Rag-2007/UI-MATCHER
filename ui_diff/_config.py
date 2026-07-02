"""
_config.py — Configuration loading and merging.
"""
from __future__ import annotations

import pathlib
from typing import Union

import yaml

# Path to the bundled default config shipped with the package
_DEFAULT_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "default_weights.yaml"


def load_config(source: Union[str, pathlib.Path, dict, None] = None) -> dict:
    """Load config from a YAML file path, a dict, or the bundled default.

    Merges provided values on top of the defaults so callers only need to
    specify overrides.
    """
    # Load bundled defaults first
    with open(_DEFAULT_CONFIG, "r") as f:
        defaults = yaml.safe_load(f)

    if source is None:
        return defaults

    if isinstance(source, dict):
        return _deep_merge(defaults, source)

    path = pathlib.Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(defaults, user_cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
