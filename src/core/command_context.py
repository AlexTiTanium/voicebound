from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Sequence, TypeVar

from loguru import logger

from core.task_runner import TaskHooks, TaskRunner, TaskSpec
from utils import configure_logging, load_config
from utils.command_utils import ProgressReporter, ProviderSettings, load_provider_settings

if TYPE_CHECKING:
    from core.summary_reporter import SummaryReporter

T = TypeVar("T")


@dataclass
class CommandContext:
    """
    Container for shared command configuration and provider settings.

    Attributes:
        config: Loaded configuration dictionary (user/config supplied).
        provider: ProviderSettings resolved from config.
        logger: Logger instance used by commands.

    Example:
        >>> ctx = CommandContext(
        ...     config={},
        ...     provider=ProviderSettings(
        ...         api_key="sk-...",
        ...         model="gpt-5-nano",
        ...         rpm=60,
        ...         concurrency=4,
        ...         retry=RetryConfig(),
        ...     ),
        ... )
    """

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
    """
    Initialize command context with configuration and logging.

    Args:
        config_path: Path to the configuration file.
        provider_key: Key for the provider in the config (e.g., "openai").
        default_model: Default model to use if not specified in config.
        default_rpm: Default requests per minute.
        concurrency_override: Optional override for concurrency.
        log_level: Logging level (e.g., "INFO", "DEBUG").
        color: Whether to enable colored logging.

    Returns:
        A CommandContext object containing the loaded config and provider settings.
    """
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


def build_tasks(worklist: Iterable[tuple[str, Callable[[], Awaitable[T]]]]) -> list[TaskSpec[T]]:
    """Convert key→coro_factory pairs into TaskSpec list."""
    specs: list[TaskSpec[T]] = []
    for key, coro_factory in worklist:

        async def wrapper(fn=coro_factory):
            """Invoke the task coroutine factory and await its result."""
            return await fn()

        specs.append(TaskSpec(task_id=key, coro_factory=wrapper))
    return specs


async def run_with_progress(
    name: str,
    total: int,
    runner: TaskRunner[T],
    specs: Sequence[TaskSpec[T]],
    summary: SummaryReporter,
    success_cb: Callable[[TaskSpec[T], T], None] | None = None,
    failure_cb: Callable[[TaskSpec[T], BaseException], None] | None = None,
    retry_cb: Callable[[TaskSpec[T], int, float | None], None] | None = None,
) -> SummaryReporter:
    """
    Run a batch of tasks with a progress bar and reporting hooks.

    Args:
        name: Name of the operation (for the progress bar).
        total: Total number of tasks.
        runner: The TaskRunner instance to use.
        specs: List of TaskSpec objects.
        summary: SummaryReporter to update.
        success_cb: Optional callback for success.
        failure_cb: Optional callback for failure.
        retry_cb: Optional callback for retries.

    Returns:
        The updated SummaryReporter.
    """
    progress = ProgressReporter(f"[{name.upper()}] Processing", total=total)

    async def hook_success(spec: TaskSpec[T], result: T):
        """
        Handle a successful task completion.

        Args:
            spec: Task specification that completed.
            result: Result value produced by the task.
        """
        if success_cb:
            success_cb(spec, result)
        else:
            summary.record_success(spec.task_id)
        progress.advance()
        return result

    async def hook_failure(spec: TaskSpec[T], exc: BaseException):
        """
        Handle a failed task and update progress.

        Args:
            spec: Task specification that failed.
            exc: Exception raised by the task.
        """
        if failure_cb:
            failure_cb(spec, exc)
        else:
            summary.record_failure(spec.task_id, exc)
        progress.advance()

    def retry_hook(spec: TaskSpec[T], attempt: int, sleep_for: float | None) -> None:
        """
        Log or delegate retry events.

        Args:
            spec: Task specification being retried.
            attempt: Current attempt count.
            sleep_for: Seconds to wait before retrying.
        """
        if retry_cb:
            retry_cb(spec, attempt, sleep_for)
        else:
            sleep_desc = f"{sleep_for:.2f}s" if sleep_for is not None else "unknown"
            logger.warning(
                f"[{name.upper()}] {spec.task_id} retry {attempt}/{runner.config.retry.attempts} "
                f"(sleep {sleep_desc})"
            )

    runner.hooks = TaskHooks[T](
        on_success=hook_success,
        on_failure=hook_failure,
        on_retry=retry_hook,
    )

    with progress:
        await runner.run(specs)
    return summary
