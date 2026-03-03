"""
config_loader.py
Loads and merges YAML config files. All modules import from here.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache

# Load .env once at import time
load_dotenv()

# Project root = directory containing this file's grandparent (src/utils/ -> src/ -> finsight/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=None)
def load_config(config_path: str = None) -> dict:
    """
    Load config/settings.yaml (and optionally a second config file).
    Returns merged dict. Results are cached — call invalidate_config_cache() to reload.
    """
    base_path = config_path or (PROJECT_ROOT / "config" / "settings.yaml")
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve all relative paths to absolute, anchored at project root
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            if val and not os.path.isabs(val):
                cfg["paths"][key] = str(PROJECT_ROOT / val)

    return cfg


@lru_cache(maxsize=None)
def load_chunking_config(experiment: str = "default") -> dict:
    """Load chunking.yaml and return the named experiment block."""
    path = PROJECT_ROOT / "config" / "chunking.yaml"
    with open(path, "r", encoding="utf-8") as f:
        all_experiments = yaml.safe_load(f)
    if experiment not in all_experiments:
        raise ValueError(
            f"Chunking experiment '{experiment}' not found. "
            f"Available: {list(all_experiments.keys())}"
        )
    return all_experiments[experiment]


@lru_cache(maxsize=None)
def load_prompts() -> dict:
    """Load config/prompts.yaml."""
    path = PROJECT_ROOT / "config" / "prompts.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def invalidate_config_cache():
    """Clear all cached configs (useful in tests or after config changes)."""
    load_config.cache_clear()
    load_chunking_config.cache_clear()
    load_prompts.cache_clear()


def get_project_root() -> Path:
    return PROJECT_ROOT
