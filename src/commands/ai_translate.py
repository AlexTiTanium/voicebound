from pathlib import Path
from threading import Lock
from typing import Iterable, TypedDict

import anyio
import typer
from loguru import logger

from apis.translation_api import (
    TranslationFilters,
    TranslationProgress,
    TranslationResult,
    TranslationService,
    TranslationSettings,
)
from core.summary_reporter import SummaryReporter
from providers.registry import get_translation_provider_info
from utils import (
    PROJECT_ROOT,
    compile_regex,
    configure_logging,
    ensure_directory,
    get_config_value,
    load_config,
    resolve_path,
)
from utils.command_utils import load_progress, load_provider_settings, load_strings


class Summary(TypedDict):
    """
    Aggregate counts for translation outcomes.

    Attributes:
        translated: Count of successfully translated strings.
        skipped: Count of strings skipped due to filters.
        loaded: Count of strings loaded from cache.
        ignored: Count of strings ignored by regex.
        empty: Count of empty strings.
        errors: List of keys that failed translation.

    Example:
        >>> summary: Summary = {
        ...     "translated": 3,
        ...     "skipped": 1,
        ...     "loaded": 2,
        ...     "ignored": 0,
        ...     "empty": 0,
        ...     "errors": [],
        ... }
    """

    translated: int
    skipped: int
    loaded: int
    ignored: int
    empty: int
    errors: list[str]


def translate_strings(
    input_file: Path | None = None,
    output_file: Path | None = None,
    progress_file: Path | None = None,
    *,
    allowed_regex: str | None = None,
    ignore_regex: str | None = None,
    dry_run: bool | None = None,
    model: str | None = None,
    count_tokens_enabled: bool | None = None,
    target_language: str | None = None,
    config_path: Path | None = PROJECT_ROOT / "config.toml",
    log_level: str | None = None,
    color: bool = True,
) -> None:
    """
    Translate strings.xml using configured provider according to config and CLI overrides.

    Orchestrates the translation process:
    1. Loads configuration and provider settings.
    2. Parses the input XML file.
    3. Filters strings based on regex patterns.
    4. Checks against cached progress.
    5. Executes translation tasks concurrently.
    6. Updates the XML file and progress cache.

    Args:
        input_file: Path to the input strings.xml file.
        output_file: Path to save the translated XML file.
        progress_file: Path to the progress JSON cache file.
        allowed_regex: Regex to select keys for translation.
        ignore_regex: Regex to exclude keys from translation.
        dry_run: If True, simulate the process without API calls.
        model: Model identifier to use.
        count_tokens_enabled: Whether to count tokens (requires tiktoken).
        target_language: Target language for translation.
        config_path: Path to the configuration file.
        log_level: Logging verbosity.
        color: Enable colored logging.
    """
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    translate_cfg = config.get("translate", {})
    provider = get_config_value(config, "translate", "provider", required=False, default="openai")
    provider_info = get_translation_provider_info(str(provider))
    if provider_info is None:
        logger.warning(
            f"[TRANSLATE] Provider '{provider}' is not recognized; defaulting to OpenAI client."
        )
        provider_info = get_translation_provider_info("openai")
    if provider_info is None:
        raise SystemExit("[TRANSLATE] No translation providers are registered.")

    provider_settings = load_provider_settings(
        config,
        provider_key=provider_info.key,
        default_model=provider_info.default_model,
        default_rpm=provider_info.default_rpm,
    )
    api_key = provider_settings.api_key
    model = model or provider_settings.model
    provider_client = provider_info.factory(api_key)

    input_file = resolve_path(input_file or translate_cfg.get("input_file", "strings.xml"))
    output_file = resolve_path(
        output_file or translate_cfg.get("output_file", "out/values/strings.xml")
    )
    progress_file = resolve_path(
        progress_file or translate_cfg.get("progress_file", ".cache/progress.json")
    )
    allowed_regex = allowed_regex or translate_cfg.get("allowed_regex", r"^chp10_")
    ignore_regex = ignore_regex or translate_cfg.get("ignore_regex", r"app_name")
    dry_run = translate_cfg.get("dry_run", False) if dry_run is None else dry_run
    count_tokens_enabled = (
        translate_cfg.get("count_tokens_enabled", True)
        if count_tokens_enabled is None
        else count_tokens_enabled
    )
    target_language = target_language or translate_cfg.get("target_language", "Russian")

    translate_pattern = compile_regex(allowed_regex, label="allowed")
    ignore_pattern = compile_regex(ignore_regex, label="ignore")

    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")

    ensure_directory(output_file.parent)
    ensure_directory(progress_file.parent)

    tree, root = load_strings(input_file)

    done: dict[str, str | None] = load_progress(progress_file, default={}) or {}
    progress_lock = Lock()
    tasks = list(root.findall("string"))
    translation_filters = TranslationFilters(
        allowed_pattern=translate_pattern,
        ignore_pattern=ignore_pattern,
    )
    translation_settings = TranslationSettings(
        model=model,
        target_language=target_language,
        dry_run=dry_run,
        count_tokens_enabled=count_tokens_enabled,
    )
    translation_progress = TranslationProgress(
        done=done,
        progress_file=progress_file,
        progress_lock=progress_lock,
    )
    service = TranslationService(provider_client, provider_settings)
    pre_results, translate_nodes = service.prepare_nodes(
        tasks,
        filters=translation_filters,
        done=done,
    )

    if not dry_run:
        logger.info(f"[TRANSLATE] Translating {len(translate_nodes)} strings using {model}.")
    else:
        logger.info(f"[TRANSLATE] Dry run enabled for {len(translate_nodes)} strings.")

    try:
        summary = SummaryReporter("translate")
        for name, _, status in pre_results:
            if isinstance(status, str):
                summary.record_translation(status, name)

        if translate_nodes:
            async def _run_translate() -> list[TranslationResult]:
                """
                Execute the async translation task runner.

                Returns:
                    A list of TranslationResult tuples from the provider tasks.
                """
                return await service.translate_nodes_async(
                    translate_nodes,
                    filters=translation_filters,
                    progress=translation_progress,
                    settings=translation_settings,
                    summary=summary,
                )

            results = anyio.run(_run_translate)
            results = pre_results + results
        else:
            summary.log_translation(str(progress_file))
            results = pre_results
    except KeyboardInterrupt:
        logger.warning("[TRANSLATE] Interrupted by user.")
        raise SystemExit(130)

    if dry_run:
        _print_dry_run(results)
        return

    service.apply_translations(root, results)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def _print_dry_run(results: Iterable[TranslationResult]) -> None:
    """Print/log a dry-run summary showing counts and token estimates."""
    total_tokens = 0
    count = 0

    for name, data, status in results:
        match status:
            case ("dry-run", tokens, preview):
                total_tokens += tokens
                count += 1
                logger.info(f"[DRY] {name}: {tokens} tokens → '{preview}...'")

    logger.info("=== SUMMARY ===")
    logger.info(f"Strings to be translated: {count}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Estimated input cost (~${total_tokens/1_000_000 * 0.005:.4f})")
    logger.info(f"Estimated output cost (~${total_tokens/1_000_000 * 0.40:.4f})")
    logger.info("No translation performed (dry-run mode).")


def typer_command(
    ctx: typer.Context,
    input_file: Path | None = typer.Option(None, help="Path to the input strings.xml."),
    output_file: Path | None = typer.Option(None, help="Path to write translated XML."),
    allowed_regex: str | None = typer.Option(
        None, help="Translate only entries matching this regex."
    ),
    ignore_regex: str | None = typer.Option(None, help="Ignore entries matching this regex."),
    dry_run: bool | None = typer.Option(None, help="Dry run (no translation calls)."),
    model: str | None = typer.Option(None, help="OpenAI model to use."),
    target_language: str | None = typer.Option(None, help="Target language to translate into."),
    config_path: Path | None = typer.Option(None, help="Path to config.toml."),
) -> None:
    """Typer CLI wrapper for translate_strings."""
    obj = ctx.ensure_object(dict)
    cfg_raw = config_path or obj.get("config_path")
    cfg_path = Path(cfg_raw) if cfg_raw else None
    log_level = obj.get("log_level")
    color = obj.get("color", True)
    translate_strings(
        input_file=input_file,
        output_file=output_file,
        allowed_regex=allowed_regex,
        ignore_regex=ignore_regex,
        dry_run=dry_run,
        model=model,
        target_language=target_language,
        config_path=cfg_path,
        log_level=log_level,
        color=color,
    )


def _summarize_results(results: Iterable[TranslationResult]) -> Summary:
    """
    Aggregate translation results into a summary dictionary.

    Args:
        results: Iterable of translation results.

    Returns:
        A Summary dictionary containing counts and error lists.
    """
    summary: Summary = {
        "translated": 0,
        "skipped": 0,
        "loaded": 0,
        "ignored": 0,
        "empty": 0,
        "errors": [],
    }
    for name, _text, status in results:
        if status == "translated":
            summary["translated"] += 1
        elif status == "skipped":
            summary["skipped"] += 1
        elif status == "loaded":
            summary["loaded"] += 1
        elif status == "ignored":
            summary["ignored"] += 1
        elif status == "empty":
            summary["empty"] += 1
        elif isinstance(status, tuple) and status[0] == "error":
            if name:
                summary["errors"].append(name)
    return summary


if __name__ == "__main__":
    typer.run(typer_command)
