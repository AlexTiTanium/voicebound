import time
from pathlib import Path
from typing import Any, Dict, Optional

import anyio
import httpx
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

from task_runner import (
    RateLimitConfig,
    RetryConfig,
    RunnerConfig,
    TaskHooks,
    TaskRunner,
    TaskSpec,
)
from utils import (
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


def build_runner_config(
    config: dict,
    voice_cfg: dict,
    concurrency_override: int | None = None,
    jitter_override: float | None = None,
) -> RunnerConfig:
    """Construct RunnerConfig from Hume provider limits."""
    hume_cfg = config.get("hume_ai", {})
    rate_cfg = hume_cfg.get("rate_limit", {})
    retry_cfg = hume_cfg.get("retry", {})

    # Fallback to legacy fields if new config is absent.
    request_delay = voice_cfg.get("request_delay_seconds", 5.5)
    backoff_seq = voice_cfg.get("backoff_seconds", [1, 2, 4])

    concurrency = concurrency_override or hume_cfg.get("concurrency") or voice_cfg.get("max_workers", 4)
    rate_limit = RateLimitConfig(
        max_per_interval=int(rate_cfg.get("max_per_interval", 1)),
        interval_seconds=float(rate_cfg.get("interval_seconds", request_delay)),
    )
    retry = RetryConfig(
        attempts=int(retry_cfg.get("attempts", voice_cfg.get("max_retries", 3))),
        backoff_base=float(retry_cfg.get("backoff_base", backoff_seq[0] if backoff_seq else 1)),
        backoff_max=float(retry_cfg.get("backoff_max", backoff_seq[-1] if backoff_seq else 4)),
        jitter=bool(
            retry_cfg.get(
                "jitter", True if jitter_override is None else jitter_override > 0
            )
        ),
    )
    return RunnerConfig(
        name="voice",
        concurrency=int(concurrency),
        rate_limit=rate_limit,
        retry=retry,
    )


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
    """Generate voice files from cached translations using Hume."""
    configure_logging(level=log_level, color=color)
    config = load_config(config_path)
    api_key = get_config_value(config, "hume_ai", "api_key")
    voice_cfg = config.get("voice", {})
    hume_cfg = config.get("hume_ai", {})

    model = get_config_value(config, "hume_ai", "model", required=False, default="octave")
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
    jitter_fraction = jitter_fraction if jitter_fraction is not None else voice_cfg.get("jitter_fraction", 0.1)
    max_elapsed_seconds = (
        max_elapsed_seconds if max_elapsed_seconds is not None else voice_cfg.get("max_elapsed_seconds", None)
    )

    if not input_file.exists():
        raise SystemExit(f"Progress file not found: {input_file}. Run translate first.")

    progress = load_json(input_file, default={})
    ensure_directory(output_dir)
    headers = {
        "Content-Type": "application/json",
        "X-Hume-Api-Key": api_key,
    }
    existing_outputs = {
        path.stem for path in output_dir.glob(f"*.{audio_format}")
    } if output_dir.exists() else set()

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

    runner_cfg = build_runner_config(
        config,
        voice_cfg,
        concurrency_override=voice_cfg.get("max_workers"),
        jitter_override=jitter_fraction,
    )
    try:
        results = anyio.run(
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
            runner_cfg,
        )
    except KeyboardInterrupt:
        logger.warning("[VOICE] Interrupted by user.")
        raise SystemExit(130)

    successes = [name for name, status in results if status == "ok"]
    failures = [name for name, status in results if status == "error"]

    logger.info(
        f"[VOICE] Run complete. generated={len(successes)} failures={len(failures)} "
        f"skipped={len(existing_outputs)}"
    )
    if failures:
        logger.error(f"[VOICE] Failed entries: {', '.join(failures)}")


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
    runner_cfg: RunnerConfig,
) -> list[tuple[str, str]]:
    """Execute Hume synthesis tasks via TaskRunner."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=True,
    )
    results: list[tuple[str, str]] = []

    async def on_success(spec: TaskSpec, _result: Path) -> None:
        results.append((spec.task_id, "ok"))
        progress.update(task_id, advance=1)

    async def on_failure(spec: TaskSpec, exc: BaseException) -> None:
        logger.error(f"[VOICE] {spec.task_id} failed: {exc}")
        results.append((spec.task_id, "error"))
        progress.update(task_id, advance=1)

    def on_retry(spec: TaskSpec, attempt: int, sleep_for: float | None) -> None:
        sleep_desc = f"{sleep_for:.2f}s" if sleep_for is not None else "unknown"
        logger.warning(
            f"[VOICE] {spec.task_id} retry {attempt}/{runner_cfg.retry.attempts} "
            f"(sleep {sleep_desc})"
        )

    runner = TaskRunner(
        runner_cfg, TaskHooks(on_success=on_success, on_failure=on_failure, on_retry=on_retry)
    )
    specs: list[TaskSpec] = []
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
            specs.append(
                TaskSpec(
                    task_id=key,
                    payload=payload,
                    coro_factory=lambda payload=payload, out_path=out_path: synthesize_once(
                        client=client,
                        headers=headers,
                        payload=payload,
                        out_path=out_path,
                        max_elapsed_seconds=max_elapsed_seconds,
                    ),
                )
            )

        with progress:
            task_id = progress.add_task("[VOICE] Processing", total=len(specs))
            await runner.run(specs)

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
