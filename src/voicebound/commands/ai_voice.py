from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

import requests
import typer
from loguru import logger

from voicebound.utils import (
    PROJECT_ROOT,
    RateLimiter,
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
) -> Dict[str, Any]:
    """Construct request payload for a single utterance."""
    return {
        "model": model,
        "format": {"type": audio_format},
        "split_utterances": split_utterances,
        "version": "2",
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
) -> None:
    """Process a single progress entry end-to-end (thread-safe)."""
    logger.info(f"[VOICE] {key} start")
    payload = build_payload(
        text,
        model=model,
        voice_name=voice_name,
        provider=provider,
        audio_format=audio_format,
        split_utterances=split_utterances,
    )
    success, error_message = attempt_send(
        payload=payload,
        headers=headers,
        out_path=out_path,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
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
) -> tuple[bool, str]:
    """Handle retries, backoff, and file write for one payload; obeys rate limit."""
    attempt = 0
    success = False
    error_message = ""

    while attempt < max_retries and not success:
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
        except requests.RequestException as exc:
            error_message = str(exc)

        if not success and attempt < max_retries:
            sleep_for = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
            logger.warning(f"[VOICE] {out_path.stem} retry after {sleep_for}s")
            time.sleep(sleep_for)

    return success, error_message


def generate_voice(
    input_file: Path | None = None,
    output_dir: Path | None = None,
    *,
    voice_name: str | None = None,
    provider: str | None = None,
    audio_format: str | None = None,
    split_utterances: bool | None = None,
    allowed_regex: str | None = None,
    ignore_regex: str | None = None,
    stop_after: int | None = None,
    max_workers: int | None = None,
    request_delay_seconds: float | None = None,
    max_retries: int | None = None,
    backoff_seconds: tuple[int, ...] | None = None,
    model: str | None = None,
    target_language: str | None = None,
    config_path: Path = PROJECT_ROOT / "config.toml",
) -> None:
    """Generate voice files from cached translations."""
    configure_logging()
    config = load_config(config_path)
    api_key = get_config_value(config, "hume_ai", "api_key")
    model = model or get_config_value(config, "hume_ai", "model", required=False, default="octave")
    provider = provider or get_config_value(config, "hume_ai", "provider", required=False, default="HUME_AI")
    voice_name = voice_name or get_config_value(config, "hume_ai", "voice_name", required=False, default="ivan")
    target_language = target_language or config.get("voice", {}).get("target_language", "Russian")
    voice_cfg = config.get("voice", {})

    input_file = resolve_path(input_file or voice_cfg.get("input_file", ".cache/progress.json"))
    output_dir = resolve_path(output_dir or voice_cfg.get("output_dir", "out/hume"))
    audio_format = audio_format or voice_cfg.get("audio_format", "mp3")
    split_utterances = voice_cfg.get("split_utterances", True) if split_utterances is None else split_utterances
    allowed_regex = allowed_regex or voice_cfg.get("allowed_regex", r"^chp")
    ignore_regex = ignore_regex or voice_cfg.get("ignore_regex", r"")
    stop_after = voice_cfg.get("stop_after", 0) if stop_after is None else stop_after
    max_workers = voice_cfg.get("max_workers", 4) if max_workers is None else max_workers
    request_delay_seconds = (
        voice_cfg.get("request_delay_seconds", 5.5) if request_delay_seconds is None else request_delay_seconds
    )
    max_retries = voice_cfg.get("max_retries", 3) if max_retries is None else max_retries
    backoff_seconds = tuple(voice_cfg.get("backoff_seconds", [1, 2, 4])) if backoff_seconds is None else backoff_seconds
    target_language = target_language or voice_cfg.get("target_language", "Russian")

    if not input_file.exists():
        raise SystemExit(f"Progress file not found: {input_file}. Run translate first.")

    progress = load_json(input_file, default={})
    ensure_directory(output_dir)
    headers = {
        "Content-Type": "application/json",
        "X-Hume-Api-Key": api_key,
    }
    rate_limiter = RateLimiter(request_delay_seconds)
    existing_outputs = {path.stem for path in output_dir.glob("*.mp3")} if output_dir.exists() else set()

    logger.info(f"[VOICE] Found {len(existing_outputs)} existing outputs; skipping those keys.")
    worklist: list[tuple[str, str]] = []
    allowed_pattern = re.compile(allowed_regex)
    ignore_pattern = re.compile(ignore_regex) if ignore_regex else None

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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                )
            )
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - safeguard
                logger.error(f"[VOICE] Unhandled exception: {exc}")

    logger.info("[VOICE] Run complete.")


def typer_command(
    input_file: Path | None = typer.Option(None, help="Path to cached progress JSON."),
    output_dir: Path | None = typer.Option(None, help="Directory to write audio files."),
    voice_name: str | None = typer.Option(None, help="Voice to request from provider."),
    provider: str | None = typer.Option(None, help="Voice provider identifier."),
    audio_format: str | None = typer.Option(None, help="Audio format extension and API format type."),
    split_utterances: bool | None = typer.Option(None, help="Whether to split utterances in the API."),
    allowed_regex: str | None = typer.Option(None, help="Only process keys matching this regex."),
    ignore_regex: str | None = typer.Option(None, help="Skip keys matching this regex."),
    stop_after: int | None = typer.Option(None, help="Stop after N items (0 for no limit)."),
    max_workers: int | None = typer.Option(None, help="Parallel workers."),
    request_delay_seconds: float | None = typer.Option(None, help="Delay between requests to honor rate limits."),
    max_retries: int | None = typer.Option(None, help="Retry attempts per item."),
    model: str | None = typer.Option(None, help="Hume AI TTS model to use."),
    target_language: str | None = typer.Option(None, help="Target language for voice content (metadata only)."),
    config_path: Path = typer.Option(PROJECT_ROOT / "config.toml", help="Path to config.toml."),
) -> None:
    """Typer-friendly wrapper."""
    generate_voice(
        input_file=input_file,
        output_dir=output_dir,
        voice_name=voice_name,
        provider=provider,
        audio_format=audio_format,
        split_utterances=split_utterances,
        allowed_regex=allowed_regex,
        ignore_regex=ignore_regex,
        stop_after=stop_after,
        max_workers=max_workers,
        request_delay_seconds=request_delay_seconds,
        max_retries=max_retries,
        backoff_seconds=None,
        model=model,
        target_language=target_language,
        config_path=config_path,
    )


if __name__ == "__main__":
    typer.run(typer_command)
