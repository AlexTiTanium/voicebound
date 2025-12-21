from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import anyio
import httpx
from loguru import logger

from core.command_context import run_with_progress
from core.task_runner import TaskHooks, TaskSpec
from utils.command_utils import ProviderSettings, build_runner, build_task_specs, log_retry

API_URL = "https://api.hume.ai/v0/tts/file"

VoiceResult = tuple[str, str]


@dataclass(frozen=True)
class VoiceSettings:
    model: str
    voice_name: str
    provider: str
    audio_format: str
    split_utterances: bool
    octave_version: str
    max_elapsed_seconds: float | None


class VoiceService:
    """Reusable voice synthesis API for cached translations."""

    def __init__(self, provider_settings: ProviderSettings | None = None, api_url: str = API_URL):
        self._provider_settings = provider_settings
        self._api_url = api_url

    def build_payload(self, text: str, *, settings: VoiceSettings) -> Dict[str, Any]:
        """Construct the Hume request payload for one utterance."""
        return {
            "model": settings.model,
            "format": {"type": settings.audio_format},
            "split_utterances": settings.split_utterances,
            "version": settings.octave_version,
            "utterances": [
                {
                    "text": text,
                    "voice": {"name": settings.voice_name, "provider": settings.provider},
                }
            ],
        }

    async def send_request(
        self, client: httpx.AsyncClient, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> httpx.Response:
        """POST to Hume TTS endpoint using an async client."""
        return await client.post(self._api_url, headers=headers, json=payload, timeout=120)

    async def synthesize_once(
        self,
        *,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        out_path: Path,
        max_elapsed_seconds: float | None,
    ) -> Path:
        """Send one synthesis request and persist the audio file."""
        start = time.perf_counter()
        response = await self.send_request(client, headers, payload)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        if max_elapsed_seconds is not None and (time.perf_counter() - start) > max_elapsed_seconds:
            raise TimeoutError(f"max elapsed {max_elapsed_seconds}s exceeded")
        await anyio.to_thread.run_sync(out_path.write_bytes, response.content)
        return out_path

    async def run_voice_async(
        self,
        worklist: list[tuple[str, str]],
        *,
        headers: dict[str, str],
        output_dir: Path,
        settings: VoiceSettings,
        summary,
        skipped_count: int,
    ) -> list[VoiceResult]:
        """Execute Hume synthesis tasks via TaskRunner."""
        provider_settings = self._provider_settings
        if provider_settings is None:
            raise ValueError("provider_settings is required for run_voice_async.")
        results: list[VoiceResult] = []
        runner = build_runner("voice", provider_settings, TaskHooks())
        work_items: list[tuple[str, Any]] = []
        async with httpx.AsyncClient() as client:
            for key, text in worklist:
                out_path = output_dir / f"{key}.{settings.audio_format}"
                payload = self.build_payload(text, settings=settings)

                async def coro(payload=payload, out_path=out_path):
                    return await self.synthesize_once(
                        client=client,
                        headers=headers,
                        payload=payload,
                        out_path=out_path,
                        max_elapsed_seconds=settings.max_elapsed_seconds,
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
                log_retry(
                    "voice",
                    spec.task_id,
                    attempt,
                    provider_settings.retry.attempts,
                    sleep_for,
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
