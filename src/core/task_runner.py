from __future__ import annotations

from dataclasses import dataclass
from inspect import iscoroutine
from typing import Any, Awaitable, Callable, Generic, Iterable, Optional, TypeVar

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
    attempts: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 10.0
    jitter: bool = True


@dataclass
class RunnerConfig:
    name: str
    rpm: int
    concurrency: int
    retry: RetryConfig


@dataclass
class TaskSpec(Generic[T]):
    task_id: str
    coro_factory: TaskFunc[T]


@dataclass
class TaskOutcome(Generic[T]):
    task_id: str
    ok: bool
    result: T | None
    error: BaseException | None
    attempts: int


@dataclass
class TaskHooks(Generic[T]):
    on_success: Optional[Callable[[TaskSpec[T], T], Awaitable[None] | None]] = None
    on_failure: Optional[Callable[[TaskSpec[T], BaseException], Awaitable[None] | None]] = None
    on_retry: Optional[Callable[[TaskSpec[T], int, float | None], None]] = None


class TaskRunner(Generic[T]):
    """Shared async runner with rpm-based rate limit, concurrency, and retries."""

    def __init__(self, config: RunnerConfig, hooks: TaskHooks[T] | None = None):
        self.config = config
        self.hooks = hooks or TaskHooks()
        self._limiter = AsyncLimiter(max_rate=config.rpm, time_period=60)
        self._semaphore = anyio.Semaphore(config.concurrency)

    async def run(self, tasks: Iterable[TaskSpec[T]]) -> list[TaskOutcome[T]]:
        outcomes: list[TaskOutcome[T]] = []
        lock = anyio.Lock()

        async with anyio.create_task_group() as tg:
            for spec in tasks:
                tg.start_soon(self._run_single, spec, outcomes, lock)

        return outcomes

    async def _run_single(
        self, spec: TaskSpec[T], outcomes: list[TaskOutcome[T]], lock: anyio.Lock
    ) -> None:
        attempts = 0
        result: T | None = None
        retrying = self._make_retrying(spec)

        try:
            async for attempt in retrying:
                attempts = attempt.retry_state.attempt_number
                async with self._semaphore:
                    async with self._limiter:
                        with attempt:
                            result = await spec.coro_factory()
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
                attempts=attempts or self.config.retry.attempts,
            )
            await self._record_outcome(outcomes, lock, outcome)
            await self._maybe_call(self.hooks.on_failure, spec, err)
        except Exception as exc:  # pragma: no cover - defensive
            outcome = TaskOutcome[T](
                task_id=spec.task_id,
                ok=False,
                result=None,
                error=exc,
                attempts=attempts or 1,
            )
            await self._record_outcome(outcomes, lock, outcome)
            await self._maybe_call(self.hooks.on_failure, spec, exc)

    def _make_retrying(self, spec: TaskSpec[T]) -> AsyncRetrying:
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
        def handler(state: RetryCallState) -> None:
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
        async with lock:
            outcomes.append(outcome)

    @staticmethod
    async def _maybe_call(callback: Callable[..., Any] | None, *args: Any) -> None:
        if not callback:
            return
        result = callback(*args)
        if iscoroutine(result):
            await result
