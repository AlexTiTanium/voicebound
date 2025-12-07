from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any, Dict, Optional

import requests
import typer
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from utils import (
    RateLimiter,
    compile_regex,
    configure_logging,
    ensure_directory,
    get_config_value,
    load_config,
    load_json,
    resolve_path,
)

API_URL = "https://api.hume.ai/v0/tts/file"


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


def send_request(headers: Dict[str, str], payload: Dict[str, Any]) -> requests.Response:
    """POST to Hume TTS endpoint."""
    return requests.post(API_URL, headers=headers, json=payload, timeout=120)


def handle_entry(
    *,
    key: str,
    text: str,
    headers: Dict[str, str],
    rate_limiter: RateLimiter,
    out_path: Path,
    max_retries: int,
    backoff_seconds: tuple[int, ...],
    model: str,
    voice_name: str,
    provider: str,
    audio_format: str,
    split_utterances: bool,
    target_language: str,
    octave_version: str,
    jitter_fraction: float,
    max_elapsed_seconds: float | None,
    stop_event: Event,
) -> None:
    """Process one entry end-to-end: build payload, send, and persist result.

    Args:
        key: Progress key for naming output.
        text: Text to synthesize.
        headers: HTTP headers including API key.
        rate_limiter: Shared rate limiter instance.
        out_path: Destination file path.
        max_retries: How many retries to attempt.
        backoff_seconds: Backoff schedule between retries.
        model: TTS model to use.
        voice_name: Voice name.
        provider: Provider identifier.
        audio_format: Audio format string.
        split_utterances: Whether to split utterances server-side.
        target_language: Informational target language string.
        octave_version: API version for Octave.
        stop_event: Event used to cancel in-flight work.
    """
    logger.info(f"[VOICE] {key} start")
    payload = build_payload(
        text,
        model=model,
        voice_name=voice_name,
        provider=provider,
        audio_format=audio_format,
        split_utterances=split_utterances,
        octave_version=octave_version,
    )
    success, error_message = attempt_send(
        payload=payload,
        headers=headers,
        out_path=out_path,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        jitter_fraction=jitter_fraction,
        max_elapsed_seconds=max_elapsed_seconds,
        stop_event=stop_event,
    )

    if not success:
        logger.error(f"[VOICE] {key} error: {error_message}")
    else:
        logger.info(f"[VOICE] {key} done")


def attempt_send(
    *,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    out_path: Path,
    rate_limiter: RateLimiter,
    max_retries: int,
    backoff_seconds: tuple[int, ...],
    jitter_fraction: float,
    max_elapsed_seconds: float | None,
    stop_event: Event,
) -> tuple[bool, str]:
    """Handle retries, backoff, and file write for one payload; obeys rate limit."""
    attempt = 0
    success = False
    error_message = ""
    start = time.perf_counter()

    while attempt < max_retries and not success:
        if stop_event.is_set():
            return False, "interrupted"
        logger.debug(f"[VOICE] {out_path.stem} attempt {attempt + 1}")
        rate_limiter.wait()
        attempt += 1
        try:
            response = send_request(headers, payload)
            if response.status_code == 200:
                out_path.write_bytes(response.content)
                success = True
            else:
                error_message = f"HTTP {response.status_code}: {response.text}"
        except KeyboardInterrupt:
            logger.warning(f"[VOICE] {out_path.stem} interrupted during attempt {attempt}")
            raise
        except requests.RequestException as exc:
            error_message = str(exc)

        if not success and attempt < max_retries:
            base_sleep = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
            jitter = base_sleep * jitter_fraction
            sleep_for = max(0, base_sleep + random.uniform(-jitter, jitter))
            logger.warning(f"[VOICE] {out_path.stem} retry after {sleep_for:.2f}s")
            time.sleep(sleep_for)
        if max_elapsed_seconds is not None and (time.perf_counter() - start) > max_elapsed_seconds:
            return False, f"max elapsed {max_elapsed_seconds}s exceeded"

    return success, error_message


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
    jitter_fraction: float | None = None,
    max_elapsed_seconds: float | None = None,
) -> None:
    """Generate voice files from cached translations using Hume.

    Args:
        input_file: Progress JSON path (overrides config when provided).
        output_dir: Directory to write audio outputs.
        target_language: Descriptive target language string.
        allowed_regex: Regex for allowed keys.
        ignore_regex: Regex for keys to skip.
        stop_after: Limit number of items to process.
        audio_format: Audio format extension/type.
        provider: Voice provider identifier.
        config_path: Path to config.toml.
    """
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    api_key = get_config_value(config, "hume_ai", "api_key")
    voice_cfg = config.get("voice", {})
    hume_cfg = config.get("hume_ai", {})

    model = get_config_value(config, "hume_ai", "model", required=False, default="octave")
    provider = provider or voice_cfg.get("provider", "HUME_AI")
    voice_name = get_config_value(config, "hume_ai", "voice_name", required=False, default="ivan")
    target_language = target_language or voice_cfg.get("target_language", "Russian")
    split_utterances = hume_cfg.get("split_utterances", True)

    input_file = resolve_path(input_file or voice_cfg.get("input_file", ".cache/progress.json"))
    output_dir = resolve_path(output_dir or voice_cfg.get("output_dir", "out/hume"))
    audio_format = audio_format or voice_cfg.get("audio_format", "mp3")
    allowed_regex = allowed_regex or voice_cfg.get("allowed_regex", r"^chp")
    ignore_regex = ignore_regex or voice_cfg.get("ignore_regex", r"")
    stop_after = voice_cfg.get("stop_after", 0) if stop_after is None else stop_after
    max_workers = voice_cfg.get("max_workers", 4)
    request_delay_seconds = voice_cfg.get("request_delay_seconds", 5.5)
    max_retries = voice_cfg.get("max_retries", 3)
    backoff_seconds = tuple(voice_cfg.get("backoff_seconds", [1, 2, 4]))
    target_language = target_language or voice_cfg.get("target_language", "Russian")
    octave_version = hume_cfg.get("octave_version", "2")
    jitter_fraction = (
        jitter_fraction if jitter_fraction is not None else voice_cfg.get("jitter_fraction", 0.1)
    )
    max_elapsed_seconds = (
        max_elapsed_seconds
        if max_elapsed_seconds is not None
        else voice_cfg.get("max_elapsed_seconds", None)
    )

    stop_event = Event()
    if not input_file.exists():
        raise SystemExit(f"Progress file not found: {input_file}. Run translate first.")

    progress = load_json(input_file, default={})
    ensure_directory(output_dir)
    headers = {
        "Content-Type": "application/json",
        "X-Hume-Api-Key": api_key,
    }
    rate_limiter = RateLimiter(request_delay_seconds)
    existing_outputs = (
        {path.stem for path in output_dir.glob("*.mp3")} if output_dir.exists() else set()
    )

    logger.info(f"[VOICE] Found {len(existing_outputs)} existing outputs; skipping those keys.")
    worklist: list[tuple[str, str]] = []
    allowed_pattern = compile_regex(allowed_regex, label="allowed")
    ignore_pattern = compile_regex(ignore_regex, label="ignore") if ignore_regex else None

    for key, text in progress.items():
        if stop_after and len(worklist) >= stop_after:
            logger.info(f"[VOICE] stop_after reached at {stop_after} items.")
            break
        if key in existing_outputs:
            logger.debug(f"[VOICE] {key} skipped (already generated).")
            continue
        if ignore_pattern and ignore_pattern.match(key):
            logger.debug(f"[VOICE] {key} skipped (ignore regex).")
            continue
        if not allowed_pattern.match(key):
            logger.debug(f"[VOICE] {key} skipped (regex).")
            continue
        out_path = output_dir / f"{key}.{audio_format}"
        if out_path.exists():
            logger.debug(f"[VOICE] {key} skipped (exists).")
            continue
        worklist.append((key, text))

    logger.info(f"[VOICE] Worklist size: {len(worklist)}")

    if not worklist:
        logger.info("[VOICE] Nothing to process.")
        return

    logger.info(f"[VOICE] Starting generation with {max_workers} workers.")
    executor = ThreadPoolExecutor(max_workers=max_workers)
    interrupted = False
    failures: list[str] = []
    successes: list[str] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=True,
    )
    with progress:
        try:
            task_id = progress.add_task("[VOICE] Processing", total=len(worklist))
            futures = []
            for key, text in worklist:
                out_path = output_dir / f"{key}.{audio_format}"
                futures.append(
                    executor.submit(
                        handle_entry,
                        key=key,
                        text=text,
                        headers=headers,
                        rate_limiter=rate_limiter,
                        out_path=out_path,
                        max_retries=max_retries,
                        backoff_seconds=backoff_seconds,
                        model=model,
                        voice_name=voice_name,
                        provider=provider,
                        audio_format=audio_format,
                        split_utterances=split_utterances,
                        target_language=target_language,
                        octave_version=octave_version,
                        jitter_fraction=jitter_fraction,
                        max_elapsed_seconds=max_elapsed_seconds,
                        stop_event=stop_event,
                    )
                )
            for future, (key, _) in zip(futures, worklist):
                try:
                    future.result()
                    successes.append(key)
                except KeyboardInterrupt:
                    logger.warning("[VOICE] Interrupted; stopping remaining futures.")
                    interrupted = True
                    stop_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(130)
                except Exception as exc:  # pragma: no cover - safeguard
                    failures.append(key)
                    logger.error(f"[VOICE] Unhandled exception for {key}: {exc}")
                finally:
                    progress.update(task_id, advance=1)
        except KeyboardInterrupt:
            logger.warning("[VOICE] Interrupted by user. Cancelling pending tasks.")
            interrupted = True
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
            raise SystemExit(130)
        finally:
            if not interrupted:
                executor.shutdown(wait=True, cancel_futures=True)

    logger.info(
        f"[VOICE] Run complete. generated={len(successes)} failures={len(failures)} "
        f"skipped={len(existing_outputs)}"
    )
    if failures:
        logger.error(f"[VOICE] Failed entries: {', '.join(failures)}")


def typer_command(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, help="Path to cached progress JSON."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to write audio files."),
    provider: Optional[str] = typer.Option(None, help="Voice provider identifier."),
    target_language: Optional[str] = typer.Option(
        None, help="Target language for voice content (metadata only)."
    ),
    allowed_regex: Optional[str] = typer.Option(None, help="Only process keys matching this regex."),
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
