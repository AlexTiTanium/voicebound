from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from loguru import logger

if TYPE_CHECKING:
    from core.types import TranslationSummaryStatus


@dataclass
class SummaryReporter:
    """
    Collects and reports statistics for batch operations.

    Tracks successes, failures, skips, and other status counts during
    translation or voice generation tasks.

    Example:
        >>> reporter = SummaryReporter("translate")
        >>> reporter.record_translation("translated", "chp1_hello")
    """

    name: str
    successes: int = 0
    successes_list: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    skipped: int = 0
    loaded: int = 0
    ignored: int = 0
    empty: int = 0

    def record_translation(self, status: TranslationSummaryStatus, task_id: str | None) -> None:
        """
        Record the outcome of a translation task.

        Args:
            status: One of "translated", "skipped", "loaded", "ignored", "empty", "error".
            task_id: The identifier of the task (e.g., string name).
        """
        if status == "translated":
            self.successes += 1
        elif status == "skipped":
            self.skipped += 1
        elif status == "loaded":
            self.loaded += 1
        elif status == "ignored":
            self.ignored += 1
        elif status == "empty":
            self.empty += 1
        elif status == "error" and task_id:
            self.failures.append(task_id)
        elif status == "dry-run":
            return

    def record_success(self, task_id: str | None) -> None:
        """
        Record a successful task by ID.

        Args:
            task_id: Task identifier, if available.
        """
        self.successes += 1
        if task_id:
            self.successes_list.append(task_id)

    def record_failure(self, task_id: str | None, _exc: BaseException | None = None) -> None:
        """
        Record a failed task by ID.

        Args:
            task_id: Task identifier, if available.
            _exc: Exception that caused the failure (unused).
        """
        if task_id:
            self.failures.append(task_id)

    def log_translation(self, output_path: str | None = None) -> None:
        """
        Log a summary of translation results.

        Args:
            output_path: Optional path to the output file for user visibility.
        """
        logger.success(
            f"[{self.name.upper()}] Done. translated={self.successes} skipped={self.skipped} "
            f"loaded={self.loaded} ignored={self.ignored} empty={self.empty} "
            f"errors={len(self.failures)}"
        )
        if self.failures:
            logger.error(f"[{self.name.upper()}] Failed entries: {', '.join(self.failures)}")
        if output_path:
            logger.success(f"[{self.name.upper()}] Output saved to: {output_path}")

    def log_voice(self, successes: Iterable[str], failures: Iterable[str], skipped: int) -> None:
        """
        Log a summary of voice generation results.

        Args:
            successes: Iterable of successful keys.
            failures: Iterable of failed keys.
            skipped: Count of skipped keys.
        """
        success_count = len(list(successes))
        failure_list = list(failures)
        logger.info(
            f"[{self.name.upper()}] Run complete. generated={success_count} "
            f"failures={len(failure_list)} skipped={skipped}"
        )
        if failure_list:
            logger.error(f"[{self.name.upper()}] Failed entries: {', '.join(failure_list)}")
