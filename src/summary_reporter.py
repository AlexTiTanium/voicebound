from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from loguru import logger


@dataclass
class SummaryReporter:
    name: str
    successes: int = 0
    failures: list[str] = field(default_factory=list)
    skipped: int = 0
    loaded: int = 0
    ignored: int = 0
    empty: int = 0

    def record_translation(self, status: str, task_id: str | None) -> None:
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

    def log_translation(self, output_path: str | None = None) -> None:
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
        success_count = len(list(successes))
        failure_list = list(failures)
        logger.info(
            f"[{self.name.upper()}] Run complete. generated={success_count} "
            f"failures={len(failure_list)} skipped={skipped}"
        )
        if failure_list:
            logger.error(f"[{self.name.upper()}] Failed entries: {', '.join(failure_list)}")
