from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable

import anyio
import tiktoken
from loguru import logger

from command_context import run_with_progress
from task_runner import TaskHooks, TaskSpec
from utils.command_utils import ProviderSettings, build_runner, build_task_specs, persist_progress

if TYPE_CHECKING:
    from summary_reporter import SummaryReporter

TranslationResult = tuple[str | None, str | None, str | tuple]


@dataclass(frozen=True)
class TranslationFilters:
    allowed_pattern: re.Pattern[str]
    ignore_pattern: re.Pattern[str]


@dataclass(frozen=True)
class TranslationSettings:
    model: str
    target_language: str
    dry_run: bool
    count_tokens_enabled: bool


@dataclass
class TranslationProgress:
    done: dict
    progress_file: Path
    progress_lock: Lock


def clean_text(text: str | None) -> str | None:
    """Convert escaped markers to real characters; unescape XML entities."""
    if text is None:
        return None
    text = (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )
    return unescape(text)


def count_tokens(encoding, text: str) -> int:
    """Return the token count for the given text using the provided encoding."""
    return len(encoding.encode(text))


class TranslationService:
    """Reusable translation API for string resources."""

    def __init__(self, client: Any, provider_settings: ProviderSettings | None = None):
        self._client = client
        self._provider_settings = provider_settings

    def translate_text(self, text: str, model: str, target_language: str) -> str:
        """Call the OpenAI chat completion API to translate text."""
        return translate_text(self._client, text, model, target_language)

    def prepare_nodes(
        self,
        nodes: Iterable,
        *,
        filters: TranslationFilters,
        done: dict,
    ) -> tuple[list[TranslationResult], list]:
        pre_results: list[TranslationResult] = []
        translate_nodes: list = []
        for node in nodes:
            name = node.get("name") or ""
            original = (node.text or "").strip()
            matches_regex = bool(filters.allowed_pattern.match(name))

            if filters.ignore_pattern.match(name):
                pre_results.append((name, None, "ignored"))
                continue

            if not original:
                pre_results.append((name, None, "empty"))
                continue

            if name in done:
                pre_results.append((name, clean_text(done[name]), "loaded"))
                continue

            if not matches_regex:
                pre_results.append((name, clean_text(original), "skipped"))
                continue

            translate_nodes.append(node)

        return pre_results, translate_nodes

    def apply_translations(
        self, root, results: Iterable[TranslationResult]
    ) -> None:
        """Apply translated text back to the XML tree for eligible entries."""
        for name, text, status in results:
            if status in ("translated", "loaded", "skipped"):
                text = clean_text(text)
                for node in root.findall("string"):
                    if node.get("name") == name and text is not None:
                        node.text = text

    def _process_node(
        self,
        node,
        *,
        filters: TranslationFilters,
        progress: TranslationProgress,
        settings: TranslationSettings,
        encoding,
    ) -> TranslationResult:
        name = node.get("name") or ""
        original = (node.text or "").strip()
        matches_regex = bool(filters.allowed_pattern.match(name))

        if filters.ignore_pattern.match(name):
            logger.debug(f"[SKIP] {name} matched ignore regex.")
            return name, None, "ignored"

        if not original:
            logger.debug(f"[SKIP] {name} has no content.")
            return name, None, "empty"

        orig_tokens = count_tokens(encoding, original) if encoding else 0

        if name in progress.done:
            suffix = "" if matches_regex else " (bypassing translate regex)"
            logger.info(f"[CACHE] {name} loaded{suffix}.")
            translated = clean_text(progress.done[name])
            return name, translated, "loaded"

        if not matches_regex:
            logger.debug(f"[SKIP] {name} does not match translate regex.")
            return name, clean_text(original), "skipped"

        if settings.dry_run:
            return name, None, ("dry-run", orig_tokens, original[:80])

        if orig_tokens:
            logger.info(f"[TRANSLATE] {name} — {orig_tokens} tokens.")
        else:
            logger.info(f"[TRANSLATE] {name}.")

        try:
            translated = self.translate_text(original, settings.model, settings.target_language)
        except Exception as exc:  # pragma: no cover - API failure surface
            logger.error(f"[ERROR] {name} translation failed: {exc}")
            return name, None, ("error", str(exc))
        translated = clean_text(translated)

        with progress.progress_lock:
            progress.done[name] = translated
            persist_progress(progress.progress_file, progress.done)

        logger.info(f"[SAVED] {name} progress updated in {progress.progress_file}.")
        return name, translated, "translated"

    async def translate_nodes_async(
        self,
        nodes: list,
        *,
        filters: TranslationFilters,
        progress: TranslationProgress,
        settings: TranslationSettings,
        summary: SummaryReporter,
    ) -> list[TranslationResult]:
        """Drive TaskRunner for translation tasks."""
        results: list[TranslationResult] = []
        if self._provider_settings is None:
            raise ValueError("provider_settings is required for translate_nodes_async.")
        runner = build_runner(
            "translate",
            self._provider_settings,
            TaskHooks(),
        )
        encoding = (
            tiktoken.get_encoding("o200k_base") if settings.count_tokens_enabled else None
        )
        work_items: list[tuple[str, Callable[..., Any]]] = []
        for idx, node in enumerate(nodes):
            task_name = node.get("name") or f"string-{idx}"
            task_fn = functools.partial(
                self._process_node,
                node,
                filters=filters,
                progress=progress,
                settings=settings,
                encoding=encoding,
            )

            async def coro(fn=task_fn):
                return await anyio.to_thread.run_sync(fn)

            work_items.append((task_name, coro))

        specs = build_task_specs(work_items)

        def success_cb(spec: TaskSpec, result):
            results.append(result)
            status = result[2]
            if isinstance(status, tuple):
                summary.record_translation(status[0], result[0])
            else:
                summary.record_translation(status, result[0])

        def failure_cb(spec: TaskSpec, exc: BaseException):
            results.append((spec.task_id, None, ("error", str(exc))))
            summary.record_translation("error", spec.task_id)

        await run_with_progress(
            "translate",
            len(specs),
            runner,
            specs,
            summary,
            success_cb=success_cb,
            failure_cb=failure_cb,
        )
        summary.log_translation(str(progress.progress_file))
        return results


def translate_text(client: Any, text: str, model: str, target_language: str) -> str:
    """Call the OpenAI chat completion API to translate text."""
    prompt = f"""
Translate the following text into {target_language} in a literary, artistic manner.
Do not add anything, do not modify structure, only translate the meaning:

{text}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def process_string(
    node,
    *,
    translate_pattern: re.Pattern[str],
    ignore_pattern: re.Pattern[str],
    done: dict,
    progress_lock: Lock,
    client: Any,
    progress_file: Path,
    model: str,
    dry_run: bool,
    encoding,
    target_language: str,
) -> TranslationResult:
    """Translate one string node with caching, filters, and progress updates."""
    filters = TranslationFilters(
        allowed_pattern=translate_pattern,
        ignore_pattern=ignore_pattern,
    )
    settings = TranslationSettings(
        model=model,
        target_language=target_language,
        dry_run=dry_run,
        count_tokens_enabled=bool(encoding),
    )
    progress = TranslationProgress(
        done=done,
        progress_file=progress_file,
        progress_lock=progress_lock,
    )
    service = TranslationService(client)
    return service._process_node(
        node,
        filters=filters,
        progress=progress,
        settings=settings,
        encoding=encoding,
    )
