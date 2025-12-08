from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Sequence

from loguru import logger

from command_utils import ProgressReporter, ProviderSettings, load_provider_settings
from task_runner import TaskHooks, TaskRunner, TaskSpec
from utils import configure_logging, load_config

if TYPE_CHECKING:
    from summary_reporter import SummaryReporter


@dataclass
class CommandContext:
    config: dict
    provider: ProviderSettings
    logger: Any = logger


def make_command_context(
    *,
    config_path: Path | None,
    provider_key: str,
    default_model: str,
    default_rpm: int,
    concurrency_override: int | None = None,
    log_level: str | None = None,
    color: bool = True,
) -> CommandContext:
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    provider = load_provider_settings(
        config,
        provider_key=provider_key,
        default_model=default_model,
        default_rpm=default_rpm,
        concurrency_override=concurrency_override,
    )
    return CommandContext(config=config, provider=provider, logger=logger)


def build_tasks(worklist: Iterable[tuple[str, Callable[[], Awaitable]]]) -> list[TaskSpec]:
    """Convert key→coro_factory pairs into TaskSpec list."""
    specs: list[TaskSpec] = []
    for key, coro_factory in worklist:
        async def wrapper(fn=coro_factory):
            return await fn()

        specs.append(TaskSpec(task_id=key, coro_factory=wrapper))
    return specs


async def run_with_progress(
    name: str,
    total: int,
    runner: TaskRunner,
    specs: Sequence[TaskSpec],
    summary: SummaryReporter,
    success_cb: Callable[[TaskSpec, Any], None] | None = None,
    failure_cb: Callable[[TaskSpec, BaseException], None] | None = None,
    retry_cb: Callable[[TaskSpec, int, float | None], None] | None = None,
) -> SummaryReporter:
    progress = ProgressReporter(f"[{name.upper()}] Processing", total=total)

    async def hook_success(spec: TaskSpec, result):
        if success_cb:
            success_cb(spec, result)
        else:
            summary.record_success(spec.task_id)
        progress.advance()
        return result

    async def hook_failure(spec: TaskSpec, exc: BaseException):
        if failure_cb:
            failure_cb(spec, exc)
        else:
            summary.record_failure(spec.task_id, exc)
        progress.advance()

    def retry_hook(spec: TaskSpec, attempt: int, sleep_for: float | None) -> None:
        if retry_cb:
            retry_cb(spec, attempt, sleep_for)
        else:
            sleep_desc = f"{sleep_for:.2f}s" if sleep_for is not None else "unknown"
            logger.warning(
                f"[{name.upper()}] {spec.task_id} retry {attempt}/{runner.config.retry.attempts} "
                f"(sleep {sleep_desc})"
            )

    runner.hooks = TaskHooks(
        on_success=hook_success,
        on_failure=hook_failure,
        on_retry=retry_hook,
    )

    with progress:
        await runner.run(specs)
    return summary
