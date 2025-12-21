from pathlib import Path
from threading import Lock
from typing import Iterable, Optional, TypedDict, cast

import anyio
import typer
from loguru import logger
from openai import OpenAI

from apis.translation_api import (
    OpenAIClient,
    TranslationFilters,
    TranslationProgress,
    TranslationResult,
    TranslationService,
    TranslationSettings,
)
from core.command_context import CommandContext, make_command_context
from core.summary_reporter import SummaryReporter
from utils import (
    PROJECT_ROOT,
    compile_regex,
    ensure_directory,
    get_config_value,
    resolve_path,
)
from utils.command_utils import load_progress, load_strings


class Summary(TypedDict):
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
    max_workers: int | None = None,
    model: str | None = None,
    count_tokens_enabled: bool | None = None,
    target_language: str | None = None,
    config_path: Path | None = PROJECT_ROOT / "config.toml",
    log_level: str | None = None,
    color: bool = True,
) -> None:
    """Translate strings.xml using OpenAI according to config and CLI overrides."""
    ctx: CommandContext = make_command_context(
        config_path=config_path,
        provider_key="openai",
        default_model="gpt-5-nano",
        default_rpm=60,
        concurrency_override=max_workers,
        log_level=log_level,
        color=color,
    )
    api_key = ctx.provider.api_key
    model = model or ctx.provider.model
    translate_cfg = ctx.config.get("translate", {})
    provider = get_config_value(
        ctx.config, "translate", "provider", required=False, default="openai"
    )
    if str(provider).lower() != "openai":
        logger.warning(
            f"[TRANSLATE] Provider '{provider}' is not recognized; defaulting to OpenAI client."
        )

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
    max_workers = translate_cfg.get("max_workers", 20) if max_workers is None else max_workers
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
    client = cast(OpenAIClient, OpenAI(api_key=api_key))
    service = TranslationService(client, ctx.provider)
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
        async def _run_translate() -> list[TranslationResult]:
            return await service.translate_nodes_async(
                translate_nodes,
                filters=translation_filters,
                progress=translation_progress,
                settings=translation_settings,
                summary=summary,
            )

        results = anyio.run(_run_translate)
        results = pre_results + results
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
    input_file: Optional[Path] = typer.Option(None, help="Path to the input strings.xml."),
    output_file: Optional[Path] = typer.Option(None, help="Path to write translated XML."),
    allowed_regex: Optional[str] = typer.Option(
        None, help="Translate only entries matching this regex."
    ),
    ignore_regex: Optional[str] = typer.Option(None, help="Ignore entries matching this regex."),
    dry_run: Optional[bool] = typer.Option(None, help="Dry run (no translation calls)."),
    max_workers: Optional[int] = typer.Option(None, help="Parallel workers."),
    model: Optional[str] = typer.Option(None, help="OpenAI model to use."),
    target_language: Optional[str] = typer.Option(None, help="Target language to translate into."),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.toml."),
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
        max_workers=max_workers,
        model=model,
        target_language=target_language,
        config_path=cfg_path,
        log_level=log_level,
        color=color,
    )


def _summarize_results(results: Iterable[TranslationResult]) -> Summary:
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
