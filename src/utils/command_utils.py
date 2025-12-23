from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    TypeAlias,
    TypeVar,
    cast,
)

from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from core.task_runner import RetryConfig, RunnerConfig, TaskHooks, TaskRunner, TaskSpec
from utils import get_config_value, load_json, write_json

T = TypeVar("T")
if TYPE_CHECKING:
    ElementTreeT: TypeAlias = ET.ElementTree[ET.Element | None]
    from core.types import AudioFormat, ProviderKey
else:
    ElementTreeT = ET.ElementTree


@dataclass
class ProviderSettings:
    """
    Configuration for a provider runtime session.

    Attributes:
        api_key: User-supplied secret used to authenticate API calls.
        model: User/config-selected model identifier.
        rpm: Requests-per-minute rate limit from user/config.
        concurrency: Derived or user-specified concurrency cap.
        retry: Retry configuration derived from config.

    Example:
        >>> settings = ProviderSettings(
        ...     api_key="sk-...",
        ...     model="gpt-5-nano",
        ...     rpm=60,
        ...     concurrency=4,
        ...     retry=RetryConfig(),
        ... )
    """

    api_key: str
    model: str
    rpm: int
    concurrency: int
    retry: RetryConfig


def derive_concurrency(rpm: int, override: int | None = None) -> int:
    """
    Compute a reasonable concurrency to approach the rpm target.

    Args:
        rpm: Requests per minute target.
        override: Optional manual concurrency override.

    Returns:
        An integer representing the number of concurrent workers.
    """
    if override:
        return max(1, int(override))
    cpus = os.cpu_count() or 4
    est = max(1, (rpm + 29) // 30)
    return max(1, min(cpus, est))


def load_retry_defaults(config: Mapping[str, Any]) -> RetryConfig:
    """
    Load retry settings from the top-level config.

    Args:
        config: Full configuration mapping (user-supplied).

    Returns:
        A RetryConfig instance populated from config defaults.
    """
    retry_cfg = config.get("retry", {})
    return RetryConfig(
        attempts=int(retry_cfg.get("attempts", 3)),
        backoff_base=float(retry_cfg.get("backoff_base", 0.5)),
        backoff_max=float(retry_cfg.get("backoff_max", 8.0)),
        jitter=bool(retry_cfg.get("jitter", True)),
    )


def load_provider_settings(
    config: Mapping[str, Any],
    *,
    provider_key: ProviderKey,
    default_model: str,
    default_rpm: int,
) -> ProviderSettings:
    """
    Load and validate provider settings from configuration.

    Args:
        config: The full configuration dictionary.
        provider_key: The key for the provider in the config.
        default_model: Default model if not specified.
        default_rpm: Default RPM if not specified.

    Returns:
        A ProviderSettings object.
    """
    provider_cfg = config.get(provider_key, {})
    api_key = get_config_value(config, provider_key, "api_key")
    model = get_config_value(config, provider_key, "model", required=False, default=default_model)
    rpm = int(get_config_value(config, provider_key, "rpm", required=False, default=default_rpm))
    retry = load_retry_defaults(config)
    concurrency = derive_concurrency(rpm, provider_cfg.get("concurrency"))
    return ProviderSettings(
        api_key=api_key,
        model=model,
        rpm=rpm,
        concurrency=concurrency,
        retry=retry,
    )


def build_runner(name: str, settings: ProviderSettings, hooks: TaskHooks[T]) -> TaskRunner[T]:
    """
    Create a configured TaskRunner instance.

    Args:
        name: Name of the runner (for logging).
        settings: Provider settings (RPM, concurrency, retry).
        hooks: Task hooks for success/failure/retry.

    Returns:
        A configured TaskRunner.
    """
    runner_cfg = RunnerConfig(
        name=name,
        rpm=settings.rpm,
        concurrency=settings.concurrency,
        retry=settings.retry,
    )
    return TaskRunner(runner_cfg, hooks)


def build_task_specs(
    worklist: Iterable[tuple[str, Callable[[], Awaitable[T]]]],
) -> list[TaskSpec[T]]:
    """
    Convert a list of (id, coroutine_factory) tuples into TaskSpec objects.

    Args:
        worklist: Iterable of (task_id, coro_factory) tuples.

    Returns:
        A list of TaskSpec objects ready for the TaskRunner.
    """
    specs: list[TaskSpec[T]] = []
    for task_id, coro_factory in worklist:

        async def wrapper(fn=coro_factory):
            """Invoke the task coroutine factory and await its result."""
            return await fn()

        specs.append(TaskSpec(task_id=task_id, coro_factory=wrapper))
    return specs


@dataclass
class OutcomeCollector:
    """
    Collects success and failure IDs from task execution.

    Useful for simple reporting where full TaskOutcome objects are not needed.

    Example:
        >>> collector = OutcomeCollector("translate")
        >>> collector.record_success("chp1_hello")
    """

    name: str
    successes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def record_success(self, task_id: str | None) -> None:
        """
        Record a successful task by its ID.

        Args:
            task_id: Task identifier emitted by the runner.
        """
        if task_id:
            self.successes.append(task_id)

    def record_failure(self, task_id: str | None, _exc: BaseException | None = None) -> None:
        """
        Record a failed task by its ID.

        Args:
            task_id: Task identifier emitted by the runner.
            _exc: Exception for the failure (unused, kept for signature parity).
        """
        if task_id:
            self.failures.append(task_id)


@dataclass
class RunnerCallbacks:
    """
    Container for task runner callbacks.

    Holds functions to be called on success, failure, and retry events.

    Example:
        >>> callbacks = RunnerCallbacks(on_success=lambda spec, result: None)
    """

    on_success: Callable[[TaskSpec[Any], Any], None] | None = None
    on_failure: Callable[[TaskSpec[Any], BaseException], None] | None = None
    on_retry: Callable[[TaskSpec[Any], int, float | None], None] | None = None


def log_retry(
    command: str, task_id: str, attempt: int, total: int, sleep_for: float | None
) -> None:
    """
    Log a retry attempt with standard formatting.

    Args:
        command: The command name (e.g., "translate").
        task_id: The ID of the task being retried.
        attempt: The current attempt number.
        total: Total allowed attempts.
        sleep_for: Seconds to sleep before next attempt.
    """
    sleep_desc = f"{sleep_for:.2f}s" if sleep_for is not None else "unknown"
    logger.warning(f"[{command.upper()}] {task_id} retry {attempt}/{total} (sleep {sleep_desc})")


def load_strings(path: Path) -> tuple[ElementTreeT, ET.Element]:
    """
    Parse an XML file and return the tree and root element.

    Args:
        path: Path to the XML file.

    Returns:
        A tuple of (ElementTree, RootElement).

    Raises:
        ValueError: If the XML root is missing.
    """
    tree = cast(ElementTreeT, ET.parse(path))
    root = tree.getroot()
    if root is None:
        raise ValueError(f"XML root missing in {path}")
    return tree, root


def persist_progress(path: Path, data: Any) -> None:
    """
    Save progress data to a JSON file.

    Args:
        path: Path to the JSON file.
        data: Data to serialize.
    """
    if isinstance(data, Mapping):
        existing = load_json(path, default=None)
        if isinstance(existing, Mapping):
            merged = dict(existing)
            merged.update(data)
            data = merged
    write_json(path, data)


def load_progress(path: Path, default: T | None = None) -> T | None:
    """
    Load progress data from a JSON file.

    Args:
        path: Path to the JSON file.
        default: Default value if file doesn't exist or is invalid.

    Returns:
        The loaded data or the default value.
    """
    return load_json(path, default=default)


def build_voice_worklist(
    progress: Mapping[str, str | None],
    allowed_pattern: re.Pattern[str],
    ignore_pattern: re.Pattern[str] | None,
    existing_outputs: set[str],
    stop_after: int,
    output_dir: Path,
    audio_format: AudioFormat,
) -> list[tuple[str, str]]:
    """
    Filter translation progress to create a worklist for voice generation.

    Args:
        progress: Dictionary of translated strings.
        allowed_pattern: Regex for allowed keys.
        ignore_pattern: Regex for ignored keys.
        existing_outputs: Set of keys that already have audio files.
        stop_after: Max number of items to process.
        output_dir: Directory to check for existing files.
        audio_format: Audio file extension.

    Returns:
        A list of (key, text) tuples to process.
    """
    worklist: list[tuple[str, str]] = []
    for key, text in progress.items():
        if text is None:
            continue
        if stop_after and len(worklist) >= stop_after:
            break
        if key in existing_outputs:
            continue
        if ignore_pattern and ignore_pattern.match(key):
            continue
        if not allowed_pattern.match(key):
            continue
        out_path = output_dir / f"{key}.{audio_format}"
        if out_path.exists():
            continue
        worklist.append((key, text))
    return worklist


class ProgressReporter:
    """
    Unified progress reporter wrapping rich.Progress; safe to use as context manager.

    Example:
        >>> with ProgressReporter("Processing", total=3) as progress:
        ...     progress.advance()
    """

    def __init__(self, description: str, total: int):
        """
        Initialize the progress reporter.

        Args:
            description: Text description of the task.
            total: Total number of steps.
        """
        self.description = description
        self.total = total
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._task_id: TaskID | None = None

    def __enter__(self) -> ProgressReporter:
        """
        Start the progress bar and return self.

        Returns:
            The active ProgressReporter instance.
        """
        self._task_id = self._progress.add_task(self.description, total=self.total)
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Tear down the progress bar on context exit.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.
        """
        self._progress.__exit__(exc_type, exc, tb)

    def advance(self, step: int = 1) -> None:
        """
        Advance the progress bar by a number of steps.

        Args:
            step: Number of increments to apply (default: 1).
        """
        if self._task_id is None:
            return
        try:
            self._progress.update(self._task_id, advance=step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[PROGRESS] update failed: {exc}")
