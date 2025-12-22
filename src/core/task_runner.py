from __future__ import annotations

from dataclasses import dataclass
from inspect import iscoroutine
from typing import Any, Awaitable, Callable, Generic, Iterable, TypeVar

import anyio
from aiolimiter import AsyncLimiter
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

T = TypeVar("T")
TaskFunc = Callable[[], Awaitable[T]]


@dataclass
class RetryConfig:
    """
    Retry policy configuration for TaskRunner.

    Attributes:
        attempts: Total attempts before giving up.
        backoff_base: Base multiplier for backoff delays.
        backoff_max: Maximum backoff delay.
        jitter: Whether to add random jitter to backoff.

    Example:
        >>> retry = RetryConfig(attempts=3, backoff_base=0.5, backoff_max=8.0, jitter=True)
    """

    attempts: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 10.0
    jitter: bool = True


@dataclass
class RunnerConfig:
    """
    Runtime configuration for TaskRunner.

    Attributes:
        name: Label for logging and progress.
        rpm: Requests-per-minute rate limit (user/config supplied).
        concurrency: Max concurrent tasks (user/config or derived).
        retry: RetryConfig for failed tasks.

    Example:
        >>> cfg = RunnerConfig(name="translate", rpm=60, concurrency=4, retry=RetryConfig())
    """

    name: str
    rpm: int
    concurrency: int
    retry: RetryConfig


@dataclass
class TaskSpec(Generic[T]):
    """
    A unit of work for TaskRunner.

    Attributes:
        task_id: Stable identifier for logging and reporting.
        coro_factory: Zero-arg coroutine factory that performs the work.

    Example:
        >>> spec = TaskSpec(task_id="item-1", coro_factory=lambda: some_async_call())
    """

    task_id: str
    coro_factory: TaskFunc[T]


@dataclass
class TaskOutcome(Generic[T]):
    """
    Result of a task execution.

    Attributes:
        task_id: Identifier of the task.
        ok: True when the task succeeded.
        result: Successful result payload (if any).
        error: Exception captured on failure (if any).
        attempts: Number of attempts used.

    Example:
        >>> outcome = TaskOutcome(task_id="item-1", ok=True, result="ok", error=None, attempts=1)
    """

    task_id: str
    ok: bool
    result: T | None
    error: BaseException | None
    attempts: int


@dataclass
class TaskHooks(Generic[T]):
    """
    Optional hooks for TaskRunner lifecycle events.

    Attributes:
        on_success: Called with (TaskSpec, result) on success.
        on_failure: Called with (TaskSpec, exception) on failure.
        on_retry: Called before a retry with (TaskSpec, attempt, sleep).

    Example:
        >>> hooks = TaskHooks(on_retry=lambda spec, attempt, sleep: None)
    """

    on_success: Callable[[TaskSpec[T], T], Awaitable[None] | None] | None = None
    on_failure: Callable[[TaskSpec[T], BaseException], Awaitable[None] | None] | None = None
    on_retry: Callable[[TaskSpec[T], int, float | None], None] | None = None


class TaskRunner(Generic[T]):
    """
    Shared async runner with rpm-based rate limit, concurrency, and retries.

    Example:
        >>> runner = TaskRunner(
        ...     RunnerConfig(name="translate", rpm=60, concurrency=4, retry=RetryConfig())
        ... )
    """

    def __init__(self, config: RunnerConfig, hooks: TaskHooks[T] | None = None):
        """
        Initialize a TaskRunner with rate limits and concurrency.

        Args:
            config: Runner configuration (user/config supplied).
            hooks: Optional callbacks for success, failure, and retry.
        """
        self.config = config
        self.hooks = hooks or TaskHooks()
        self._limiter = AsyncLimiter(max_rate=config.rpm, time_period=60)
        self._semaphore = anyio.Semaphore(config.concurrency)

    async def run(self, tasks: Iterable[TaskSpec[T]]) -> list[TaskOutcome[T]]:
        """
        Execute a batch of tasks with concurrency control and rate limiting.

        Args:
            tasks: An iterable of TaskSpec objects defining the work to be done.

        Returns:
            A list of TaskOutcome objects containing results or errors for each task.
        """
        outcomes: list[TaskOutcome[T]] = []
        lock = anyio.Lock()

        async with anyio.create_task_group() as tg:
            for spec in tasks:
                tg.start_soon(self._run_single, spec, outcomes, lock)

        return outcomes

    async def _run_single(
        self, spec: TaskSpec[T], outcomes: list[TaskOutcome[T]], lock: anyio.Lock
    ) -> None:
        """
        Execute a single task with retries and record its outcome.

        Args:
            spec: Task specification containing the coroutine factory.
            outcomes: Shared list used to collect TaskOutcome entries.
            lock: Async lock guarding access to the outcomes list.
        """
        retrying = self._make_retrying(spec)

        try:

            async def _call() -> T:
                async with self._semaphore:
                    async with self._limiter:
                        return await spec.coro_factory()

            result = await retrying(_call)
            attempts = retrying.statistics.get("attempt_number", 1)
            outcome = TaskOutcome[T](
                task_id=spec.task_id, ok=True, result=result, error=None, attempts=attempts
            )
            await self._record_outcome(outcomes, lock, outcome)
            await self._maybe_call(self.hooks.on_success, spec, result)
            return
        except RetryError as exc:
            err = exc.last_attempt.exception() or exc
            outcome = TaskOutcome[T](
                task_id=spec.task_id,
                ok=False,
                result=None,
                error=err,
                attempts=exc.last_attempt.attempt_number,
            )
            await self._record_outcome(outcomes, lock, outcome)
            await self._maybe_call(self.hooks.on_failure, spec, err)
        except Exception as exc:  # pragma: no cover - defensive
            outcome = TaskOutcome[T](
                task_id=spec.task_id,
                ok=False,
                result=None,
                error=exc,
                attempts=retrying.statistics.get("attempt_number", 1),
            )
            await self._record_outcome(outcomes, lock, outcome)
            await self._maybe_call(self.hooks.on_failure, spec, exc)

    def _make_retrying(self, spec: TaskSpec[T]) -> AsyncRetrying:
        """
        Build a tenacity retry controller for the given task.

        Args:
            spec: TaskSpec used for hook callbacks.

        Returns:
            An AsyncRetrying instance configured with backoff and hooks.
        """
        retry_cfg = self.config.retry
        wait_strategy = (
            wait_random_exponential(multiplier=retry_cfg.backoff_base, max=retry_cfg.backoff_max)
            if retry_cfg.jitter
            else wait_exponential(multiplier=retry_cfg.backoff_base, max=retry_cfg.backoff_max)
        )
        return AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(retry_cfg.attempts),
            wait=wait_strategy,
            reraise=True,
            before_sleep=self._before_sleep(spec),
        )

    def _before_sleep(self, spec: TaskSpec[T]) -> Callable[[RetryCallState], None]:
        """
        Create a hook to run before retry sleep.

        Args:
            spec: TaskSpec used to identify the task in callbacks.

        Returns:
            A callable that can be passed to tenacity's before_sleep hook.
        """

        def handler(state: RetryCallState) -> None:
            """Invoke the user retry hook with attempt metadata."""
            if not self.hooks.on_retry:
                return
            try:
                sleep_for = getattr(state.next_action, "sleep", None)
                self.hooks.on_retry(spec, state.attempt_number, sleep_for)
            except Exception as exc:  # pragma: no cover - guardrails
                logger.debug(f"[{self.config.name}] on_retry hook failed: {exc}")

        return handler

    @staticmethod
    async def _record_outcome(
        outcomes: list[TaskOutcome[T]], lock: anyio.Lock, outcome: TaskOutcome[T]
    ) -> None:
        """
        Append a TaskOutcome to the shared list in a thread-safe manner.

        Args:
            outcomes: Shared list collecting all outcomes.
            lock: Async lock to synchronize list access.
            outcome: Outcome to append.
        """
        async with lock:
            outcomes.append(outcome)

    @staticmethod
    async def _maybe_call(callback: Callable[..., Any] | None, *args: Any) -> None:
        """
        Invoke a callback that may return a coroutine.

        Args:
            callback: Optional callback to invoke.
            *args: Arguments forwarded to the callback.
        """
        if not callback:
            return
        result = callback(*args)
        if iscoroutine(result):
            await result
