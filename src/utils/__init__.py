import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, TypeVar

import toml
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_XDG_CONFIG = (
    Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicebound" / "config.toml"
)

REQUIRED_CONFIG = {
    "openai": ["api_key"],
    "hume_ai": ["api_key"],
}

T = TypeVar("T")


def configure_logging(level: str | None = None, *, color: bool = True) -> None:
    """
    Configure loguru to log to stderr once per process (keeps stdout clean for progress).

    Args:
        level: Logging level (e.g., "INFO", "DEBUG").
        color: Whether to enable colored output.
    """
    log_level_str: str = level or os.getenv("VOICEBOUND_LOG_LEVEL") or "INFO"
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"loguru\._simple_sinks",
    )
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
        if color
        else "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    logger.remove()
    logger.add(sys.stderr, level=log_level_str, format=fmt, enqueue=False)


def resolve_config_path(config_path: Path | None = None) -> Path:
    """
    Resolve config path with precedence: explicit > env > local > XDG.

    Args:
        config_path: Explicitly provided path.

    Returns:
        The resolved Path object.

    Raises:
        SystemExit: If no configuration file is found.
    """
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


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load full config file with validation.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        SystemExit: If the config file is missing or invalid.
    """
    if config_path is not None:
        if not config_path.exists():
            raise SystemExit(f"Missing config file: {config_path}.")
        path = config_path
    else:
        path = resolve_config_path(None)
    data = toml.load(path)
    validate_config(data, path)
    return data


def validate_config(config: dict[str, Any], path: Path) -> None:
    """
    Ensure required sections/keys exist and are non-empty.

    Args:
        config: The configuration dictionary.
        path: Path to the config file (for error messages).

    Raises:
        SystemExit: If required keys are missing.
    """
    missing: list[str] = []
    for section, keys in REQUIRED_CONFIG.items():
        section_data = config.get(section, {})
        for key in keys:
            if not section_data.get(key):
                missing.append(f"[{section}].{key}")
    if missing:
        raise SystemExit(f"Missing required config keys in {path}: {', '.join(missing)}.")


def get_config_value(
    config: Mapping[str, Any],
    section: str,
    key: str,
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    """
    Fetch a value from config with optional default and required enforcement.

    Args:
        config: The configuration dictionary.
        section: The section name (e.g., "openai").
        key: The key within the section (e.g., "api_key").
        required: If True, raises SystemExit if value is missing/empty.
        default: Default value if key is missing.

    Returns:
        The configuration value.

    Raises:
        SystemExit: If required is True and value is missing.
    """
    value = config.get(section, {}).get(key, default)
    if required and (value is None or value == ""):
        raise SystemExit(f"Set [{section}].{key} in config.toml.")
    return value


def ensure_directory(path: Path) -> None:
    """
    Create directory if it does not exist.

    Args:
        path: Path to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: T | None = None) -> T | None:
    """
    Load JSON content or return default when file is missing.

    Handles trailing commas by attempting a regex repair if standard parsing fails.

    Args:
        path: Path to the JSON file.
        default: Default value if file is missing or unparseable.

    Returns:
        The parsed JSON data or the default value.
    """
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning(f"[JSON] Failed to parse {path}: {exc}")
        raw = path.read_text(encoding="utf-8")
        repaired = re.sub(r",\s*(\}|\])\s*$", r"\1", raw, count=1)
        if repaired != raw:
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                logger.error(f"[JSON] Repair failed for {path}; using default.")
                return default
            write_json(path, data)
            logger.warning(f"[JSON] Repaired trailing comma in {path}.")
            return data
        return default


def write_json(path: Path, data: Any) -> None:
    """
    Persist JSON data with UTF-8 encoding.

    Writes to a temporary file first, then renames it to ensure atomicity.

    Args:
        path: Path to the JSON file.
        data: Data to serialize.
    """
    ensure_directory(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class RateLimiter:
    """
    Thread-safe rate limiter that enforces a minimum delay between calls.

    Example:
        >>> limiter = RateLimiter(min_interval=1.0)
        >>> limiter.wait()
    """

    def __init__(self, min_interval: float):
        """
        Initialize the rate limiter.

        Args:
            min_interval: Minimum seconds between calls.
        """
        self.min_interval = min_interval
        self._lock = Lock()
        self._last: float | None = None

    def wait(self) -> None:
        """Block until the minimum interval has passed since the last call."""
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
    """
    Resolve paths relative to project root when not absolute.

    Args:
        path_value: The path string or Path object.
        base: The base directory to resolve against (default: PROJECT_ROOT).

    Returns:
        The resolved absolute Path.
    """
    path = Path(path_value)
    return path if path.is_absolute() else (base / path)


def compile_regex(pattern: str, *, label: str) -> re.Pattern[str]:
    """
    Compile regex with a clear error if invalid.

    Args:
        pattern: The regex pattern string.
        label: Label for the pattern (e.g., "allowed", "ignore") for error messages.

    Returns:
        The compiled regex pattern object.

    Raises:
        SystemExit: If the regex pattern is invalid.
    """
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise SystemExit(f"Invalid {label} regex '{pattern}': {exc}") from exc
