from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Literal, Protocol
from xml.etree.ElementTree import Element

import tiktoken
from anyio import to_thread
from loguru import logger

from core.command_context import run_with_progress
from core.task_runner import TaskHooks, TaskSpec
from utils.command_utils import ProviderSettings, build_runner, build_task_specs, persist_progress

if TYPE_CHECKING:
    from core.summary_reporter import SummaryReporter
    from providers.types import TranslationProvider


class TokenEncoder(Protocol):
    """
    Protocol for token encoders used to count tokens.

    Example:
        >>> tokens = encoder.encode("hello")
    """

    def encode(self, text: str) -> list[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input text to tokenize (user content).

        Returns:
            A list of integer token IDs.
        """
        ...


StatusLiteral = Literal["ignored", "empty", "loaded", "skipped", "translated"]
DryRunStatus = tuple[Literal["dry-run"], int, str]
ErrorStatus = tuple[Literal["error"], str]
TranslationStatus = StatusLiteral | DryRunStatus | ErrorStatus
TranslationResult = tuple[str, str | None, TranslationStatus]


@dataclass(frozen=True)
class TranslationFilters:
    """
    Regex filters that control which keys are translated.

    Attributes:
        allowed_pattern: User/config-supplied allowlist regex.
        ignore_pattern: User/config-supplied ignore regex.

    Example:
        >>> filters = TranslationFilters(
        ...     allowed_pattern=re.compile(r"^chp"),
        ...     ignore_pattern=re.compile(r"app_name"),
        ... )
    """

    allowed_pattern: re.Pattern[str]
    ignore_pattern: re.Pattern[str]


@dataclass(frozen=True)
class TranslationSettings:
    """
    Settings that drive translation behavior.

    Attributes:
        model: User/config-selected model identifier.
        target_language: User/config-selected target language.
        dry_run: User-supplied flag to skip API calls.
        count_tokens_enabled: User/config flag to count tokens.

    Example:
        >>> settings = TranslationSettings(
        ...     model="gpt-5-nano",
        ...     target_language="Spanish",
        ...     dry_run=False,
        ...     count_tokens_enabled=True,
        ... )
    """

    model: str
    target_language: str
    dry_run: bool
    count_tokens_enabled: bool


@dataclass
class TranslationProgress:
    """
    Mutable progress state for translation runs.

    Attributes:
        done: Cache of already translated keys.
        progress_file: Path to the JSON cache file.
        progress_lock: Lock for safe concurrent updates.

    Example:
        >>> progress = TranslationProgress(
        ...     done={},
        ...     progress_file=Path("progress.json"),
        ...     progress_lock=Lock(),
        ... )
    """

    done: dict[str, str | None]
    progress_file: Path
    progress_lock: Lock


def clean_text(text: str | None) -> str | None:
    """Convert escaped markers to real characters; unescape XML entities."""
    if text is None:
        return None
    text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')
    return unescape(text)


def count_tokens(encoding, text: str) -> int:
    """Return the token count for the given text using the provided encoding."""
    return len(encoding.encode(text))


class TranslationService:
    """
    Reusable translation API for string resources.

    Example:
        >>> service = TranslationService(provider)
        >>> service.translate_text("Hello", "gpt-5-nano", "Spanish")
        'Hola'
    """

    def __init__(
        self, provider: TranslationProvider, provider_settings: ProviderSettings | None = None
    ):
        """
        Initialize the service with a provider implementation.

        Args:
            provider: TranslationProvider implementation.
            provider_settings: Optional runtime settings (needed for async batch).
        """
        self._provider = provider
        self._provider_settings = provider_settings

    def translate_text(self, text: str, model: str, target_language: str) -> str:
        """
        Call the provider translate API to translate text.

        Args:
            text: The text to translate.
            model: The model identifier to use.
            target_language: The target language (e.g., "Russian").

        Returns:
            The translated text string.

        Example:
            >>> service.translate_text("Hello", "gpt-4o", "Spanish")
            'Hola'
        """
        return self._provider.translate_text(text, model, target_language)

    def prepare_nodes(
        self,
        nodes: Iterable[Element],
        *,
        filters: TranslationFilters,
        done: dict[str, str | None],
    ) -> tuple[list[TranslationResult], list[Element]]:
        """
        Filter and prepare XML nodes for translation.

        Identifies which nodes need translation, which are already done (cached),
        and which should be ignored or skipped based on filters.

        Args:
            nodes: Iterable of XML Element objects (e.g. <string name="...">...</string>).
            filters: Regex patterns for allowing/ignoring keys.
            done: Dictionary of already translated keys (cache).

        Returns:
            A tuple containing:
            1. A list of `TranslationResult` tuples for items that don't need API calls
               (ignored, empty, loaded from cache, skipped).
            2. A list of `Element` objects that require translation.
        """
        pre_results: list[TranslationResult] = []
        translate_nodes: list[Element] = []
        seen_names: set[str] = set()
        for node in nodes:
            name = node.get("name") or ""
            if name in seen_names:
                logger.debug(f"[SKIP] Duplicate key encountered; skipping extra node: {name}")
                continue
            seen_names.add(name)
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

    def apply_translations(self, root: Element, results: Iterable[TranslationResult]) -> None:
        """
        Apply translated text back to the XML tree for eligible entries.

        Updates the text content of <string> elements in the XML tree with the
        translated values from the results.

        Args:
            root: The root XML Element of the strings file.
            results: Iterable of TranslationResult tuples containing (name, text, status).
        """
        for name, text, status in results:
            if status in ("translated", "loaded", "skipped"):
                text = clean_text(text)
                for node in root.findall("string"):
                    if node.get("name") == name and text is not None:
                        node.text = text

    def _process_node(
        self,
        node: Element,
        *,
        filters: TranslationFilters,
        progress: TranslationProgress,
        settings: TranslationSettings,
        encoding: TokenEncoder | None,
    ) -> TranslationResult:
        """
        Translate a single XML node with caching and filtering.

        Args:
            node: XML node containing a string resource.
            filters: Regex filters derived from user/config inputs.
            progress: Progress state used for caching and persistence.
            settings: Translation settings, including user-configured model/language.
            encoding: Optional token encoder for counting tokens.

        Returns:
            A TranslationResult tuple (name, translated_text, status).
        """
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
        nodes: list[Element],
        *,
        filters: TranslationFilters,
        progress: TranslationProgress,
        settings: TranslationSettings,
        summary: SummaryReporter,
    ) -> list[TranslationResult]:
        """
        Drive TaskRunner for translation tasks.

        Orchestrates the concurrent translation of multiple XML nodes using the
        configured provider, respecting rate limits and concurrency settings.

        Args:
            nodes: List of XML Element objects to translate.
            filters: Filters to apply (though usually pre-filtered).
            progress: Progress tracking object.
            settings: Translation settings (model, language, etc.).
            summary: Reporter for collecting statistics.

        Returns:
            A list of TranslationResult tuples.
        """
        results: list[TranslationResult] = []
        if self._provider_settings is None:
            raise ValueError("provider_settings is required for translate_nodes_async.")
        runner = build_runner(
            "translate",
            self._provider_settings,
            TaskHooks[TranslationResult](),
        )
        encoding = tiktoken.get_encoding("o200k_base") if settings.count_tokens_enabled else None
        work_items: list[tuple[str, Callable[[], Awaitable[TranslationResult]]]] = []
        for idx, node in enumerate(nodes):
            task_name = node.get("name") or f"string-{idx}"
            def task_fn(
                node: Element = node,
                filters: TranslationFilters = filters,
                progress: TranslationProgress = progress,
                settings: TranslationSettings = settings,
                encoding: TokenEncoder | None = encoding,
            ) -> TranslationResult:
                return self._process_node(
                    node,
                    filters=filters,
                    progress=progress,
                    settings=settings,
                    encoding=encoding,
                )

            def make_coro(
                fn: Callable[[], TranslationResult],
            ) -> Callable[[], Awaitable[TranslationResult]]:
                async def coro() -> TranslationResult:
                    """
                    Run the synchronous node translation in a worker thread.

                    Returns:
                        A TranslationResult tuple for the node.
                    """
                    return await to_thread.run_sync(fn)

                return coro

            work_items.append((task_name, make_coro(task_fn)))

        specs = build_task_specs(work_items)

        def success_cb(spec: TaskSpec[TranslationResult], result: TranslationResult) -> None:
            """
            Record a successful translation result.

            Args:
                spec: TaskSpec identifying the node.
                result: TranslationResult from the provider call.
            """
            results.append(result)
            status = result[2]
            if isinstance(status, tuple):
                summary.record_translation(status[0], result[0])
            else:
                summary.record_translation(status, result[0])

        def failure_cb(spec: TaskSpec[TranslationResult], exc: BaseException) -> None:
            """
            Record a failed translation result.

            Args:
                spec: TaskSpec identifying the node.
                exc: Exception raised by the provider call.
            """
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


def process_string(
    node: Element,
    *,
    translate_pattern: re.Pattern[str],
    ignore_pattern: re.Pattern[str],
    done: dict[str, str | None],
    progress_lock: Lock,
    provider: TranslationProvider,
    progress_file: Path,
    model: str,
    dry_run: bool,
    encoding: TokenEncoder | None,
    target_language: str,
) -> TranslationResult:
    """
    Translate one string node with caching, filters, and progress updates.

    This is a standalone helper that instantiates a TranslationService to process
    a single node. It handles:
    1. Checking filters (allowed/ignored regex).
    2. Checking cache (done dict).
    3. Calling the provider if needed.
    4. Updating the progress file.

    Args:
        node: The XML Element to process.
        translate_pattern: Regex for allowed keys.
        ignore_pattern: Regex for ignored keys.
        done: Dictionary of completed translations.
        progress_lock: Lock for thread-safe progress updates.
        provider: The translation provider instance.
        progress_file: Path to the progress JSON file.
        model: Model identifier.
        dry_run: If True, skips actual API calls.
        encoding: Token encoder for counting tokens (optional).
        target_language: Target language.

    Returns:
        A TranslationResult tuple (name, text, status).
    """
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
    service = TranslationService(provider)
    return service._process_node(
        node,
        filters=filters,
        progress=progress,
        settings=settings,
        encoding=encoding,
    )
