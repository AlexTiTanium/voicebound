import time
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import anyio
import httpx
import typer
from loguru import logger

from core.command_context import run_with_progress
from core.summary_reporter import SummaryReporter
from core.task_runner import TaskHooks, TaskSpec
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
    build_runner,
    build_task_specs,
    build_voice_worklist,
    load_progress,
    load_provider_settings,
)

API_URL = "https://api.hume.ai/v0/tts/file"


class VoiceSummary(TypedDict):
    successes: list[str]
    failures: list[str]
    skipped: int


def build_payload(
    text: str,
    *,
    model: str,
    voice_name: str,
    provider: str,
    audio_format: str,
    split_utterances: bool,
    octave_version: str,
) -> Dict[str, Any]:
    """Construct the Hume request payload for one utterance.

    Args:
        text: Utterance content.
        model: TTS model name.
        voice_name: Voice to request.
        provider: Voice provider identifier.
        audio_format: Audio format string (e.g., mp3).
        split_utterances: Whether to split utterances.
        octave_version: API version to target.
    """
    return {
        "model": model,
        "format": {"type": audio_format},
        "split_utterances": split_utterances,
        "version": octave_version,
        "utterances": [
            {
                "text": text,
                "voice": {"name": voice_name, "provider": provider},
            }
        ],
    }


async def send_request(
    client: httpx.AsyncClient, headers: Dict[str, str], payload: Dict[str, Any]
) -> httpx.Response:
    """POST to Hume TTS endpoint using an async client."""
    return await client.post(API_URL, headers=headers, json=payload, timeout=120)


async def synthesize_once(
    *,
    client: httpx.AsyncClient,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    out_path: Path,
    max_elapsed_seconds: float | None,
) -> Path:
    """Send one synthesis request and persist the audio file."""
    start = time.perf_counter()
    response = await send_request(client, headers, payload)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    if max_elapsed_seconds is not None and (time.perf_counter() - start) > max_elapsed_seconds:
        raise TimeoutError(f"max elapsed {max_elapsed_seconds}s exceeded")
    await anyio.to_thread.run_sync(out_path.write_bytes, response.content)
    return out_path


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
    """Generate voice files from cached translations using Hume."""
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    voice_cfg = config.get("voice", {})
    hume_cfg = config.get("hume_ai", {})
    provider_settings = load_provider_settings(
        config,
        provider_key="hume_ai",
        default_model="octave",
        default_rpm=10,
        concurrency_override=voice_cfg.get("max_workers"),
    )
    api_key = provider_settings.api_key

    model = get_config_value(
        config, "hume_ai", "model", required=False, default=provider_settings.model
    )
    provider = provider or voice_cfg.get("provider", "HUME_AI")
    voice_name = get_config_value(config, "hume_ai", "voice_name", required=False, default="ivan")
    _target_language = target_language or voice_cfg.get("target_language", "Russian")
    split_utterances = hume_cfg.get("split_utterances", True)

    input_file = resolve_path(input_file or voice_cfg.get("input_file", ".cache/progress.json"))
    output_dir = resolve_path(output_dir or voice_cfg.get("output_dir", "out/hume"))
    audio_format = audio_format or voice_cfg.get("audio_format", "mp3")
    allowed_regex = allowed_regex or voice_cfg.get("allowed_regex", r"^chp")
    ignore_regex = ignore_regex or voice_cfg.get("ignore_regex", r"")
    stop_after = voice_cfg.get("stop_after", 0) if stop_after is None else stop_after
    octave_version = hume_cfg.get("octave_version", "2")
    max_elapsed_seconds = (
        max_elapsed_seconds
        if max_elapsed_seconds is not None
        else voice_cfg.get("max_elapsed_seconds", None)
    )

    if not input_file.exists():
        raise SystemExit(f"Progress file not found: {input_file}. Run translate first.")

    progress = load_progress(input_file, default={})
    ensure_directory(output_dir)
    headers = {
        "Content-Type": "application/json",
        "X-Hume-Api-Key": api_key,
    }
    existing_outputs = {
        path.stem for path in output_dir.glob(f"*.{audio_format}")
    } if output_dir.exists() else set()

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
            headers,
            output_dir,
            audio_format,
            model,
            voice_name,
            provider,
            split_utterances,
            octave_version,
            max_elapsed_seconds,
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
    headers: dict[str, str],
    output_dir: Path,
    audio_format: str,
    model: str,
    voice_name: str,
    provider: str,
    split_utterances: bool,
    octave_version: str,
    max_elapsed_seconds: float | None,
    provider_settings: ProviderSettings,
    summary: SummaryReporter,
    skipped_count: int,
) -> list[tuple[str, str]]:
    """Execute Hume synthesis tasks via TaskRunner."""
    results: list[tuple[str, str]] = []

    runner = build_runner("voice", provider_settings, TaskHooks())
    work_items: list[tuple[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for key, text in worklist:
            out_path = output_dir / f"{key}.{audio_format}"
            payload = build_payload(
                text,
                model=model,
                voice_name=voice_name,
                provider=provider,
                audio_format=audio_format,
                split_utterances=split_utterances,
                octave_version=octave_version,
            )

            async def coro(payload=payload, out_path=out_path):
                return await synthesize_once(
                    client=client,
                    headers=headers,
                    payload=payload,
                    out_path=out_path,
                    max_elapsed_seconds=max_elapsed_seconds,
                )

            work_items.append((key, coro))

        def success_cb(spec: TaskSpec, _result: Path) -> None:
            results.append((spec.task_id, "ok"))
            summary.record_success(spec.task_id)

        def failure_cb(spec: TaskSpec, exc: BaseException) -> None:
            logger.error(f"[VOICE] {spec.task_id} failed: {exc}")
            results.append((spec.task_id, "error"))
            summary.record_failure(spec.task_id, exc)

        def retry_cb(spec: TaskSpec, attempt: int, sleep_for: float | None) -> None:
            sleep_desc = f"{sleep_for:.2f}s" if sleep_for is not None else "unknown"
            logger.warning(
                f"[VOICE] {spec.task_id} retry {attempt}/{provider_settings.retry.attempts} "
                f"(sleep {sleep_desc})"
            )

        await run_with_progress(
            "voice",
            len(work_items),
            runner,
            build_task_specs(work_items),
            summary,
            success_cb=success_cb,
            failure_cb=failure_cb,
            retry_cb=retry_cb,
        )

    summary.skipped = skipped_count
    return results


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
