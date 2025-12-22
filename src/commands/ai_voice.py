from pathlib import Path
from typing import TypedDict

import anyio
import typer
from loguru import logger

from apis.voice_api import VoiceResult, VoiceService, VoiceSettings
from core.summary_reporter import SummaryReporter
from providers.registry import get_voice_provider_info
from providers.types import VoiceProvider
from utils import (
    compile_regex,
    configure_logging,
    ensure_directory,
    get_config_value,
    load_config,
    resolve_path,
)
from utils.command_utils import (
    ProviderSettings,
    build_voice_worklist,
    load_progress,
    load_provider_settings,
)


class VoiceSummary(TypedDict):
    """
    Summary of voice generation outcomes.

    Attributes:
        successes: Keys that successfully generated audio.
        failures: Keys that failed voice generation.
        skipped: Count of items skipped because output already existed.

    Example:
        >>> summary: VoiceSummary = {"successes": ["chp1_hello"], "failures": [], "skipped": 2}
    """

    successes: list[str]
    failures: list[str]
    skipped: int


class VoiceDryRunSummary(TypedDict):
    """
    Summary of estimated voice generation cost for dry runs.

    Attributes:
        total_chars: Total characters in the worklist.
        free_chars: Characters included in the current free allotment.
        billable_chars: Characters beyond the included amount.
        rate_per_1k: Price per 1,000 characters.
        estimated_cost: Estimated cost in the configured currency.
        currency: Currency code for display.
    """

    total_chars: int
    free_chars: int
    billable_chars: int
    rate_per_1k: float
    estimated_cost: float
    currency: str


VOICE_PRICING_DEFAULT_FREE_CHARS = 0
VOICE_PRICING_DEFAULT_RATE = 0.15


def generate_voice(
    input_file: Path | None = None,
    output_dir: Path | None = None,
    *,
    target_language: str | None = None,
    allowed_regex: str | None = None,
    ignore_regex: str | None = None,
    stop_after: int | None = None,
    audio_format: str | None = None,
    provider: str | None = None,
    config_path: Path | None = None,
    log_level: str | None = None,
    color: bool = True,
    max_elapsed_seconds: float | None = None,
    dry_run: bool | None = None,
) -> None:
    """
    Generate voice files from cached translations using configured provider.

    Orchestrates the voice generation process:
    1. Loads configuration and provider settings.
    2. Loads the progress cache (translated strings).
    3. Filters items based on regex and existing files.
    4. Executes voice synthesis tasks concurrently.

    Args:
        input_file: Path to the progress JSON file (source of text).
        output_dir: Directory to save generated audio files.
        target_language: Target language (metadata only).
        allowed_regex: Regex to select keys for generation.
        ignore_regex: Regex to exclude keys from generation.
        stop_after: Stop after processing N items.
        audio_format: Audio format extension (e.g., "mp3").
        provider: Voice provider identifier.
        config_path: Path to the configuration file.
        log_level: Logging verbosity.
        color: Enable colored logging.
        max_elapsed_seconds: Timeout for each generation request.
        dry_run: When True, only estimate costs and do not call the provider.
    """
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    voice_cfg = config.get("voice", {})
    voice_provider = provider or voice_cfg.get("provider", "hume_ai")
    provider_info = get_voice_provider_info(str(voice_provider))
    if provider_info is None:
        logger.warning(
            f"[VOICE] Provider '{voice_provider}' is not recognized; defaulting to Hume."
        )
        provider_info = get_voice_provider_info("hume_ai")
    if provider_info is None:
        raise SystemExit("[VOICE] No voice providers are registered.")
    provider_cfg = config.get(provider_info.key, {})
    provider_settings = load_provider_settings(
        config,
        provider_key=provider_info.key,
        default_model=provider_info.default_model,
        default_rpm=provider_info.default_rpm,
    )

    model = get_config_value(
        config, provider_info.key, "model", required=False, default=provider_settings.model
    )
    voice_name = get_config_value(
        config, provider_info.key, "voice_name", required=False, default="ivan"
    )
    split_utterances = provider_cfg.get("split_utterances", True)

    input_file = resolve_path(input_file or voice_cfg.get("input_file", ".cache/progress.json"))
    output_dir = resolve_path(output_dir or voice_cfg.get("output_dir", "out/hume"))
    audio_format = audio_format or provider_cfg.get("audio_format") or voice_cfg.get(
        "audio_format", "mp3"
    )
    allowed_regex = allowed_regex or voice_cfg.get("allowed_regex", r"^chp")
    ignore_regex = ignore_regex or voice_cfg.get("ignore_regex", r"")
    stop_after = voice_cfg.get("stop_after", 0) if stop_after is None else stop_after
    octave_version = provider_cfg.get("octave_version", "2")
    dry_run = voice_cfg.get("dry_run", False) if dry_run is None else dry_run
    max_elapsed_seconds = (
        max_elapsed_seconds
        if max_elapsed_seconds is not None
        else voice_cfg.get("max_elapsed_seconds", None)
    )

    if not input_file.exists():
        raise SystemExit(f"Progress file not found: {input_file}. Run translate first.")

    progress: dict[str, str | None] = load_progress(input_file, default={}) or {}
    ensure_directory(output_dir)
    api_provider_client = provider_info.factory()
    existing_outputs = (
        {path.stem for path in output_dir.glob(f"*.{audio_format}")}
        if output_dir.exists()
        else set()
    )

    logger.info(f"[VOICE] Found {len(existing_outputs)} existing outputs; skipping those keys.")
    allowed_pattern = compile_regex(allowed_regex, label="allowed")
    ignore_pattern = compile_regex(ignore_regex, label="ignore") if ignore_regex else None
    worklist = build_voice_worklist(
        progress,
        allowed_pattern,
        ignore_pattern,
        existing_outputs,
        stop_after,
        output_dir,
        audio_format,
    )

    logger.info(f"[VOICE] Worklist size: {len(worklist)}")

    if not worklist:
        logger.info("[VOICE] Nothing to process.")
        return
    if dry_run:
        free_chars = int(
            provider_cfg.get(
                "pricing_free_chars",
                voice_cfg.get("pricing_free_chars", VOICE_PRICING_DEFAULT_FREE_CHARS),
            )
        )
        rate_per_1k = float(
            provider_cfg.get(
                "pricing_rate_per_1k",
                voice_cfg.get("pricing_rate_per_1k", VOICE_PRICING_DEFAULT_RATE),
            )
        )
        currency = str(
            provider_cfg.get("pricing_currency", voice_cfg.get("pricing_currency", "USD"))
        )
        if provider_info.key == "elevenlabs":
            max_chars_limit = int(provider_cfg.get("max_chars_limit", 5000))
            for key, text in worklist:
                if len(text) > max_chars_limit:
                    logger.warning(
                        f"[VOICE] ElevenLabs max_chars_limit={max_chars_limit} exceeded by "
                        f"{key} ({len(text)} chars)."
                    )
        if octave_version != "2":
            logger.warning(
                "[VOICE] Default pricing assumes Octave 2; override pricing values if needed."
            )
        _print_voice_dry_run(
            worklist,
            free_chars=free_chars,
            rate_per_1k=rate_per_1k,
            currency=currency,
        )
        return
    summary: VoiceSummary = {"successes": [], "failures": [], "skipped": len(existing_outputs)}
    try:
        voice_summary = SummaryReporter("voice")
        anyio.run(
            _run_voice_async,
            worklist,
            output_dir,
            audio_format,
            model,
            voice_name,
            split_utterances,
            octave_version,
            max_elapsed_seconds,
            api_provider_client,
            provider_settings,
            voice_summary,
            summary["skipped"],
        )
    except KeyboardInterrupt:
        logger.warning("[VOICE] Interrupted by user.")
        raise SystemExit(130)

    voice_summary.log_voice(
        voice_summary.successes_list, voice_summary.failures, summary["skipped"]
    )


def _summarize_voice_dry_run(
    worklist: list[tuple[str, str]],
    *,
    free_chars: int,
    rate_per_1k: float,
    currency: str,
) -> VoiceDryRunSummary:
    total_chars = sum(len(text) for _, text in worklist)
    billable_chars = max(0, total_chars - max(free_chars, 0))
    estimated_cost = (billable_chars / 1000.0) * max(rate_per_1k, 0.0)
    return {
        "total_chars": total_chars,
        "free_chars": max(free_chars, 0),
        "billable_chars": billable_chars,
        "rate_per_1k": max(rate_per_1k, 0.0),
        "estimated_cost": estimated_cost,
        "currency": currency,
    }


def _print_voice_dry_run(
    worklist: list[tuple[str, str]],
    *,
    free_chars: int,
    rate_per_1k: float,
    currency: str,
) -> None:
    for key, text in worklist:
        preview = text[:80].replace("\n", " ")
        logger.info(f"[DRY] {key}: {len(text)} chars -> '{preview}...'")
    summary = _summarize_voice_dry_run(
        worklist,
        free_chars=free_chars,
        rate_per_1k=rate_per_1k,
        currency=currency,
    )
    logger.info("=== SUMMARY ===")
    logger.info(f"Total characters: {summary['total_chars']}")
    logger.info(f"Free characters: {summary['free_chars']}")
    logger.info(f"Billable characters: {summary['billable_chars']}")
    logger.info(f"Rate per 1,000 characters: {summary['rate_per_1k']} {summary['currency']}")
    logger.info(f"Estimated cost: {summary['estimated_cost']:.4f} {summary['currency']}")
    logger.info("No voice generation performed (dry-run mode).")


async def _run_voice_async(
    worklist: list[tuple[str, str]],
    output_dir: Path,
    audio_format: str,
    model: str,
    voice_name: str,
    split_utterances: bool,
    octave_version: str,
    max_elapsed_seconds: float | None,
    api_provider: VoiceProvider,
    provider_settings: ProviderSettings,
    summary: SummaryReporter,
    skipped_count: int,
) -> list[VoiceResult]:
    """
    Build VoiceSettings and run the async voice service.

    Args:
        worklist: List of (key, text) pairs to synthesize.
        output_dir: Directory to write audio files to.
        audio_format: Audio format extension for output files.
        model: Provider model identifier (user/config supplied).
        voice_name: Provider voice name (user/config supplied).
        split_utterances: Whether to let the API split utterances.
        octave_version: Provider-specific voice model version.
        max_elapsed_seconds: Optional per-request timeout.
        api_provider: Provider implementation instance.
        provider_settings: API key, model, RPM, and retry settings.
        summary: SummaryReporter used to collect counts.
        skipped_count: Number of keys skipped before processing.

    Returns:
        A list of (key, status) tuples.
    """
    voice_settings = VoiceSettings(
        model=model,
        voice_name=voice_name,
        audio_format=audio_format,
        split_utterances=split_utterances,
        octave_version=octave_version,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    service = VoiceService(api_provider, provider_settings=provider_settings)
    return await service.run_voice_async(
        worklist,
        output_dir=output_dir,
        settings=voice_settings,
        summary=summary,
        skipped_count=skipped_count,
    )


def typer_command(
    ctx: typer.Context,
    input_file: Path | None = typer.Option(None, help="Path to cached progress JSON."),
    output_dir: Path | None = typer.Option(None, help="Directory to write audio files."),
    provider: str | None = typer.Option(None, help="Voice provider identifier."),
    target_language: str | None = typer.Option(
        None, help="Target language for voice content (metadata only)."
    ),
    allowed_regex: str | None = typer.Option(None, help="Only process keys matching this regex."),
    ignore_regex: str | None = typer.Option(None, help="Skip keys matching this regex."),
    stop_after: int | None = typer.Option(None, help="Stop after N items (0 for no limit)."),
    audio_format: str | None = typer.Option(
        None, help="Audio format extension and API format type."
    ),
    config_path: Path | None = typer.Option(None, help="Path to config.toml."),
    dry_run: bool | None = typer.Option(None, help="Dry run (no voice generation)."),
) -> None:
    """Typer-friendly CLI wrapper for generate_voice."""
    obj = ctx.ensure_object(dict)
    cfg_raw = config_path or obj.get("config_path")
    cfg_path = Path(cfg_raw) if cfg_raw else None
    log_level = obj.get("log_level")
    color = obj.get("color", True)
    generate_voice(
        input_file=input_file,
        output_dir=output_dir,
        provider=provider,
        target_language=target_language,
        allowed_regex=allowed_regex,
        ignore_regex=ignore_regex,
        stop_after=stop_after,
        audio_format=audio_format,
        config_path=cfg_path,
        log_level=log_level,
        color=color,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(typer_command)
