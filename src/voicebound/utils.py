import json
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import toml
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def configure_logging(level: str | None = None) -> None:
    """Configure loguru to log to stdout once per process."""
    log_level = level or os.getenv("VOICEBOUND_LOG_LEVEL", "INFO")
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        enqueue=True,
    )


def load_config_value(section: str, key: str, config_path: Path = CONFIG_PATH) -> str:
    """Load a single config value, raising a user-friendly error when missing."""
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}. Populate [{section}].{key}.")

    config = toml.load(config_path)
    value = config.get(section, {}).get(key, "")
    if not value:
        raise SystemExit(f"Set {section}.{key} in {config_path}.")
    return value


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON content or return default when file is missing."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    """Persist JSON data with UTF-8 encoding."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


class RateLimiter:
    """Thread-safe rate limiter that enforces a minimum delay between calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = Lock()
        self._last: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = time.perf_counter()
            if self._last is not None:
                sleep_for = self.min_interval - (now - self._last)
                if sleep_for > 0:
                    logger.debug(f"Rate limiter sleeping {sleep_for:.2f}s")
                    time.sleep(sleep_for)
                    now = time.perf_counter()
            self._last = now

