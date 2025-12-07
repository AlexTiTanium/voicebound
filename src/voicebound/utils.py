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


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load full config file."""
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}.")
    return toml.load(config_path)


def get_config_value(config: dict, section: str, key: str, *, required: bool = True, default: Any = None) -> Any:
    """Fetch a value from config with optional default and required enforcement."""
    value = config.get(section, {}).get(key, default)
    if required and (value is None or value == ""):
        raise SystemExit(f"Set [{section}].{key} in config.toml.")
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


def resolve_path(path_value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve paths relative to project root when not absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else (base / path)
