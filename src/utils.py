"""
Utility functions for YOLO Pipe Counter.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data_config(path: str = "dataset/data.yaml") -> Dict[str, Any]:
    """Load dataset configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_directories(dirs: list):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_class_names(data_config: Dict[str, Any]) -> Dict[int, str]:
    """Return mapping of class index to class name."""
    return {int(k): v for k, v in data_config["names"].items()}
