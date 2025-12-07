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
    CONFIG_PATH,
    PROJECT_ROOT,
    RateLimiter,
    configure_logging,
    ensure_directory,
    load_config_value,
    load_json,
)

API_URL = "https://api.hume.ai/v0/tts/file"


def build_payload(
    text: str,
    *,
    model: str,
    voice_name: str,
    voice_provider: str,
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
                "voice": {"name": voice_name, "provider": voice_provider},
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
    voice_provider: str,
    audio_format: str,
    split_utterances: bool,
) -> None:
    """Process a single progress entry end-to-end (thread-safe)."""
    logger.info(f"[VOICE] {key} start")
    payload = build_payload(
        text,
        model=model,
        voice_name=voice_name,
        voice_provider=voice_provider,
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
            logger.info(f"[VOICE] {out_path.stem} retry after {sleep_for}s")
            time.sleep(sleep_for)

    return success, error_message


def generate_voice(
    input_file: Path = PROJECT_ROOT / ".cache/progress.json",
    output_dir: Path = PROJECT_ROOT / "out/hume",
    *,
    voice_name: str = "ivan",
    voice_provider: str = "HUME_AI",
    audio_format: str = "mp3",
    split_utterances: bool = True,
    name_regex: str = r"^chp",
    stop_after: int = 0,
    max_workers: int = 4,
    request_delay_seconds: float = 5.5,
    max_retries: int = 3,
    backoff_seconds: tuple[int, ...] = (1, 2, 4),
    model: str = "octave",
) -> None:
    """Generate voice files from cached translations."""
    configure_logging()
    api_key = load_config_value("hume_ai", "api_key", CONFIG_PATH)

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
    pattern = re.compile(name_regex)

    for key, text in progress.items():
        if stop_after and len(worklist) >= stop_after:
            logger.info(f"[VOICE] stop_after reached at {stop_after} items.")
            break
        if key in existing_outputs:
            logger.debug(f"[VOICE] {key} skipped (already generated).")
            continue
        if not pattern.match(key):
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
                    voice_provider=voice_provider,
                    audio_format=audio_format,
                    split_utterances=split_utterances,
                )
            )
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - safeguard
                logger.error(f"[VOICE] Unhandled exception: {exc}")

    logger.info("[VOICE] Run complete.")


def typer_command(
    input_file: Path = typer.Option(PROJECT_ROOT / ".cache/progress.json", help="Path to cached progress JSON."),
    output_dir: Path = typer.Option(PROJECT_ROOT / "out/hume", help="Directory to write audio files."),
    voice_name: str = typer.Option("ivan", help="Voice to request from provider."),
    voice_provider: str = typer.Option("HUME_AI", help="Voice provider identifier."),
    audio_format: str = typer.Option("mp3", help="Audio format extension and API format type."),
    split_utterances: bool = typer.Option(True, help="Whether to split utterances in the API."),
    name_regex: str = typer.Option(r"^chp", help="Only process keys matching this regex."),
    stop_after: int = typer.Option(0, help="Stop after N items (0 for no limit)."),
    max_workers: int = typer.Option(4, help="Parallel workers."),
    request_delay_seconds: float = typer.Option(5.5, help="Delay between requests to honor rate limits."),
    max_retries: int = typer.Option(3, help="Retry attempts per item."),
    model: str = typer.Option("octave", help="Hume AI TTS model to use."),
) -> None:
    """Typer-friendly wrapper."""
    generate_voice(
        input_file=input_file,
        output_dir=output_dir,
        voice_name=voice_name,
        voice_provider=voice_provider,
        audio_format=audio_format,
        split_utterances=split_utterances,
        name_regex=name_regex,
        stop_after=stop_after,
        max_workers=max_workers,
        request_delay_seconds=request_delay_seconds,
        max_retries=max_retries,
        backoff_seconds=(1, 2, 4),
        model=model,
    )


if __name__ == "__main__":
    typer.run(typer_command)
