from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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

from task_runner import RetryConfig, RunnerConfig, TaskHooks, TaskRunner
from utils import get_config_value


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


def load_retry_defaults(config: dict) -> RetryConfig:
    retry_cfg = config.get("retry", {})
    return RetryConfig(
        attempts=int(retry_cfg.get("attempts", 3)),
        backoff_base=float(retry_cfg.get("backoff_base", 0.5)),
        backoff_max=float(retry_cfg.get("backoff_max", 8.0)),
        jitter=bool(retry_cfg.get("jitter", True)),
    )


def load_provider_settings(
    config: dict,
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


def build_runner(name: str, settings: ProviderSettings, hooks: TaskHooks) -> TaskRunner:
    runner_cfg = RunnerConfig(
        name=name,
        rpm=settings.rpm,
        concurrency=settings.concurrency,
        retry=settings.retry,
    )
    return TaskRunner(runner_cfg, hooks)


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
