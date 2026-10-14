"""Load settings from ``config.yaml`` so models and paths are easy to change."""

from functools import lru_cache
from pathlib import Path

import yaml

# The project root is two folders above this file: src/guideline_rag/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read config.yaml once and cache it, so every module sees the same values."""
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Turn a path from config.yaml (relative to the project root) into an absolute one."""
    return PROJECT_ROOT / relative
