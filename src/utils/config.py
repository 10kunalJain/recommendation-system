"""Configuration loader for the recommendation system."""

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths against project root
    for key in ("raw_dir", "processed_dir"):
        if key in config.get("data", {}):
            config["data"][key] = str(PROJECT_ROOT / config["data"][key])

    return config


CONFIG = load_config()
