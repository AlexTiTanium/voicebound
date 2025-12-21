from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, TypedDict

import httpx
from anyio import to_thread
from loguru import logger

from core.command_context import run_with_progress
from core.task_runner import TaskHooks, TaskSpec
from utils.command_utils import ProviderSettings, build_runner, build_task_specs, log_retry

if TYPE_CHECKING:
    from core.summary_reporter import SummaryReporter
    from providers.types import VoiceProvider


class VoiceFormat(TypedDict):
    type: str


class VoiceMeta(TypedDict):
    name: str
    provider: str


class VoiceUtterance(TypedDict):
    text: str
    voice: VoiceMeta


class VoicePayload(TypedDict):
    model: str
    format: VoiceFormat
    split_utterances: bool
    version: str
    utterances: list[VoiceUtterance]


VoiceResult = tuple[str, Literal["ok", "error"]]


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

    def __init__(
        self, provider: VoiceProvider, provider_settings: ProviderSettings | None = None
    ):
        self._provider = provider
        self._provider_settings = provider_settings

    async def synthesize_once(
        self,
        *,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: VoicePayload,
        out_path: Path,
        max_elapsed_seconds: float | None,
    ) -> Path:
        """Send one synthesis request and persist the audio file."""
        start = time.perf_counter()
        response = await self._provider.send_request(client, headers, payload)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        if max_elapsed_seconds is not None and (time.perf_counter() - start) > max_elapsed_seconds:
            raise TimeoutError(f"max elapsed {max_elapsed_seconds}s exceeded")
        await to_thread.run_sync(out_path.write_bytes, response.content)
        return out_path

    async def run_voice_async(
        self,
        worklist: list[tuple[str, str]],
        *,
        output_dir: Path,
        settings: VoiceSettings,
        summary: SummaryReporter,
        skipped_count: int,
    ) -> list[VoiceResult]:
        """Execute voice synthesis tasks via TaskRunner."""
        provider_settings = self._provider_settings
        if provider_settings is None:
            raise ValueError("provider_settings is required for run_voice_async.")
        results: list[VoiceResult] = []
        runner = build_runner("voice", provider_settings, TaskHooks[Path]())
        headers = self._provider.build_headers(provider_settings)
        work_items: list[tuple[str, Callable[[], Awaitable[Path]]]] = []
        async with httpx.AsyncClient() as client:
            for key, text in worklist:
                out_path = output_dir / f"{key}.{settings.audio_format}"
                payload = self._provider.build_payload(text, settings=settings)

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
