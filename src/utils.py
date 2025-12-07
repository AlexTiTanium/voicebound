import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import toml
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_XDG_CONFIG = (
    Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicebound" / "config.toml"
)

REQUIRED_CONFIG = {
    "openai": ["api_key"],
    "hume_ai": ["api_key"],
}


def configure_logging(level: str | None = None, *, color: bool = True) -> None:
    """Configure loguru to log to stderr once per process (keeps stdout clean for progress)."""
    log_level_str: str = (level or os.getenv("VOICEBOUND_LOG_LEVEL") or "INFO")
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
        if color
        else "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    logger.remove()
    logger.add(sys.stderr, level=log_level_str, format=fmt, enqueue=False)


def resolve_config_path(config_path: Path | None = None) -> Path:
    """Resolve config path with precedence: explicit > env > local > XDG."""
    env_path = os.getenv("VOICEBOUND_CONFIG")
    candidates = [
        Path(config_path) if config_path else None,
        Path(env_path) if env_path else None,
        DEFAULT_LOCAL_CONFIG,
        DEFAULT_XDG_CONFIG,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise SystemExit(
        f"No config.toml found. Checked: {[str(c) for c in candidates if c is not None]}"
    )


def load_config(config_path: Path | None = None) -> dict:
    """Load full config file with validation."""
    if config_path is not None:
        if not config_path.exists():
            raise SystemExit(f"Missing config file: {config_path}.")
        path = config_path
    else:
        path = resolve_config_path(None)
    data = toml.load(path)
    validate_config(data, path)
    return data


def validate_config(config: dict, path: Path) -> None:
    """Ensure required sections/keys exist and are non-empty."""
    missing: list[str] = []
    for section, keys in REQUIRED_CONFIG.items():
        section_data = config.get(section, {})
        for key in keys:
            if not section_data.get(key):
                missing.append(f"[{section}].{key}")
    if missing:
        raise SystemExit(
            f"Missing required config keys in {path}: {', '.join(missing)}."
        )


def get_config_value(
    config: dict, section: str, key: str, *, required: bool = True, default: Any = None
) -> Any:
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
                    logger.debug(f"[RATE] sleeping {sleep_for:.2f}s")
                    time.sleep(sleep_for)
                    now = time.perf_counter()
            self._last = now


def resolve_path(path_value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve paths relative to project root when not absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else (base / path)


def compile_regex(pattern: str, *, label: str) -> re.Pattern[str]:
    """Compile regex with a clear error if invalid."""
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise SystemExit(f"Invalid {label} regex '{pattern}': {exc}") from exc
