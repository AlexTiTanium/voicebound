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
    Optional,
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
else:
    ElementTreeT = ET.ElementTree

@dataclass
class ProviderSettings:
    api_key: str
    model: str
    rpm: int
    concurrency: int
    retry: RetryConfig


def derive_concurrency(rpm: int, override: int | None = None) -> int:
    """Compute a reasonable concurrency to approach the rpm target."""
    if override:
        return max(1, int(override))
    cpus = os.cpu_count() or 4
    est = max(1, (rpm + 29) // 30)
    return max(1, min(cpus, est))


def load_retry_defaults(config: Mapping[str, Any]) -> RetryConfig:
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
    provider_key: str,
    default_model: str,
    default_rpm: int,
    concurrency_override: Optional[int] = None,
) -> ProviderSettings:
    provider_cfg = config.get(provider_key, {})
    api_key = get_config_value(config, provider_key, "api_key")
    model = get_config_value(config, provider_key, "model", required=False, default=default_model)
    rpm = int(get_config_value(config, provider_key, "rpm", required=False, default=default_rpm))
    retry = load_retry_defaults(config)
    concurrency = derive_concurrency(
        rpm, concurrency_override or provider_cfg.get("concurrency")
    )
    return ProviderSettings(
        api_key=api_key,
        model=model,
        rpm=rpm,
        concurrency=concurrency,
        retry=retry,
    )


def build_runner(name: str, settings: ProviderSettings, hooks: TaskHooks[T]) -> TaskRunner[T]:
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
    specs: list[TaskSpec[T]] = []
    for task_id, coro_factory in worklist:
        async def wrapper(fn=coro_factory):
            return await fn()

        specs.append(TaskSpec(task_id=task_id, coro_factory=wrapper))
    return specs


@dataclass
class OutcomeCollector:
    name: str
    successes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def record_success(self, task_id: str | None) -> None:
        if task_id:
            self.successes.append(task_id)

    def record_failure(self, task_id: str | None, _exc: BaseException | None = None) -> None:
        if task_id:
            self.failures.append(task_id)


@dataclass
class RunnerCallbacks:
    on_success: Callable[[TaskSpec[Any], Any], None] | None = None
    on_failure: Callable[[TaskSpec[Any], BaseException], None] | None = None
    on_retry: Callable[[TaskSpec[Any], int, float | None], None] | None = None


def log_retry(
    command: str, task_id: str, attempt: int, total: int, sleep_for: float | None
) -> None:
    sleep_desc = f"{sleep_for:.2f}s" if sleep_for is not None else "unknown"
    logger.warning(f"[{command.upper()}] {task_id} retry {attempt}/{total} (sleep {sleep_desc})")


def load_strings(path: Path) -> tuple[ElementTreeT, ET.Element]:
    tree = cast(ElementTreeT, ET.parse(path))
    root = tree.getroot()
    if root is None:
        raise ValueError(f"XML root missing in {path}")
    return tree, root


def persist_progress(path: Path, data: Any) -> None:
    write_json(path, data)


def load_progress(path: Path, default: T | None = None) -> T | None:
    return load_json(path, default=default)


def build_voice_worklist(
    progress: Mapping[str, str | None],
    allowed_pattern: re.Pattern[str],
    ignore_pattern: re.Pattern[str] | None,
    existing_outputs: set[str],
    stop_after: int,
    output_dir: Path,
    audio_format: str,
) -> list[tuple[str, str]]:
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
    """Unified progress reporter wrapping rich.Progress; safe to use as context manager."""

    def __init__(self, description: str, total: int):
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

    def __enter__(self) -> "ProgressReporter":
        self._task_id = self._progress.add_task(self.description, total=self.total)
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def advance(self, step: int = 1) -> None:
        if self._task_id is None:
            return
        try:
            self._progress.update(self._task_id, advance=step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[PROGRESS] update failed: {exc}")
