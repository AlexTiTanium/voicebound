from pathlib import Path
from typing import Optional, TypedDict

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
    successes: list[str]
    failures: list[str]
    skipped: int


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
) -> None:
    """Generate voice files from cached translations using configured provider."""
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    voice_cfg = config.get("voice", {})
    voice_provider = provider or voice_cfg.get("provider", "HUME_AI")
    api_provider = voice_cfg.get("api_provider") or voice_provider
    provider_info = get_voice_provider_info(str(api_provider))
    if provider_info is None:
        logger.warning(f"[VOICE] Provider '{api_provider}' is not recognized; defaulting to Hume.")
        provider_info = get_voice_provider_info("hume_ai")
    if provider_info is None:
        raise SystemExit("[VOICE] No voice providers are registered.")
    provider_cfg = config.get(provider_info.key, {})
    provider_settings = load_provider_settings(
        config,
        provider_key=provider_info.key,
        default_model=provider_info.default_model,
        default_rpm=provider_info.default_rpm,
        concurrency_override=voice_cfg.get("max_workers"),
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
    audio_format = audio_format or voice_cfg.get("audio_format", "mp3")
    allowed_regex = allowed_regex or voice_cfg.get("allowed_regex", r"^chp")
    ignore_regex = ignore_regex or voice_cfg.get("ignore_regex", r"")
    stop_after = voice_cfg.get("stop_after", 0) if stop_after is None else stop_after
    octave_version = provider_cfg.get("octave_version", "2")
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
            voice_provider,
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


async def _run_voice_async(
    worklist: list[tuple[str, str]],
    output_dir: Path,
    audio_format: str,
    model: str,
    voice_name: str,
    provider: str,
    split_utterances: bool,
    octave_version: str,
    max_elapsed_seconds: float | None,
    api_provider: VoiceProvider,
    provider_settings: ProviderSettings,
    summary: SummaryReporter,
    skipped_count: int,
) -> list[VoiceResult]:
    voice_settings = VoiceSettings(
        model=model,
        voice_name=voice_name,
        provider=provider,
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
    input_file: Optional[Path] = typer.Option(None, help="Path to cached progress JSON."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to write audio files."),
    provider: Optional[str] = typer.Option(None, help="Voice provider identifier."),
    target_language: Optional[str] = typer.Option(
        None, help="Target language for voice content (metadata only)."
    ),
    allowed_regex: Optional[str] = typer.Option(
        None, help="Only process keys matching this regex."
    ),
    ignore_regex: Optional[str] = typer.Option(None, help="Skip keys matching this regex."),
    stop_after: Optional[int] = typer.Option(None, help="Stop after N items (0 for no limit)."),
    audio_format: Optional[str] = typer.Option(
        None, help="Audio format extension and API format type."
    ),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.toml."),
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
    )


if __name__ == "__main__":
    typer.run(typer_command)
