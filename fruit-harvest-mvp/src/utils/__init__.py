"""Shared utility functions."""
from __future__ import annotations
from pathlib import Path
import yaml


def load_config(config_path: str | Path = "configs/settings.yaml") -> dict:
    """
    Load the project settings YAML and return as a dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
