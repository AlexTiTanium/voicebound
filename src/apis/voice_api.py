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
    """
    Audio format payload for the voice API.

    Attributes:
        type: Audio format identifier (user/config supplied, e.g., "mp3").

    Example:
        >>> fmt: VoiceFormat = {"type": "mp3"}
    """

    type: str


class VoiceMeta(TypedDict):
    """
    Voice metadata included in the API payload.

    Attributes:
        name: Voice name chosen by the user/config.

    Example:
        >>> meta: VoiceMeta = {"name": "ivan", "provider": "HUME_AI"}
    """

    name: str
    provider: str


class VoiceUtterance(TypedDict):
    """
    One utterance entry in the voice API payload.

    Attributes:
        text: Text to synthesize (user-provided content from translations).
        voice: Voice metadata describing provider and voice name.

    Example:
        >>> utterance: VoiceUtterance = {
        ...     "text": "Hello",
        ...     "voice": {"name": "ivan", "provider": "HUME_AI"},
        ... }
    """

    text: str
    voice: VoiceMeta


class VoicePayload(TypedDict, total=False):
    """
    Provider-specific JSON payload sent to the voice API.

    Attributes:
        model: Model identifier selected by user/config.
        format: Output audio format settings.
        split_utterances: Whether the provider should split utterances.
        version: Provider-specific model version.
        utterances: List of utterance entries to synthesize.
        text: Text to synthesize (provider-specific).
        input: Text to synthesize (OpenAI TTS).
        model_id: Provider-specific model identifier.
        voice_id: Provider-specific voice identifier.
        voice: Provider-specific voice selection (OpenAI TTS).
        response_format: Output format for OpenAI TTS.
        instructions: Optional voice direction for OpenAI TTS.

    Example:
        >>> payload: VoicePayload = {
        ...     "model": "octave",
        ...     "format": {"type": "mp3"},
        ...     "split_utterances": True,
        ...     "version": "2",
        ...     "utterances": [],
        ... }
    """

    model: str
    format: VoiceFormat
    split_utterances: bool
    version: str
    utterances: list[VoiceUtterance]
    text: str
    input: str
    model_id: str
    voice_id: str
    voice: str
    response_format: str
    instructions: str


VoiceResult = tuple[str, Literal["ok", "error"]]


@dataclass(frozen=True)
class VoiceSettings:
    """
    Runtime settings for voice synthesis.

    Attributes:
        model: User/config-selected voice model.
        voice_name: User/config-selected voice name.
        audio_format: Output audio format extension.
        split_utterances: Whether to let the provider split utterances.
        octave_version: Provider-specific model version.
        max_elapsed_seconds: Optional request timeout.
        enabled_acting_instruction: Whether to generate acting instructions.
        acting_instruction_model: Model to use when generating acting instructions.

    Example:
        >>> settings = VoiceSettings(
        ...     model="octave",
        ...     voice_name="ivan",
        ...     audio_format="mp3",
        ...     split_utterances=True,
        ...     octave_version="2",
        ...     max_elapsed_seconds=None,
        ...     enabled_acting_instruction=False,
        ...     acting_instruction_model="gpt-5-nano",
        ... )
    """

    model: str
    voice_name: str
    audio_format: str
    split_utterances: bool
    octave_version: str
    max_elapsed_seconds: float | None
    enabled_acting_instruction: bool = False
    acting_instruction_model: str = "gpt-5-nano"


class VoiceService:
    """
    Reusable voice synthesis API for cached translations.

    Example:
        >>> service = VoiceService(provider)
    """

    def __init__(self, provider: VoiceProvider, provider_settings: ProviderSettings | None = None):
        """
        Initialize the service with a provider implementation.

        Args:
            provider: VoiceProvider implementation.
            provider_settings: Optional runtime settings (needed for async batch).
        """
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
        """
        Send one synthesis request and persist the audio file.

        Args:
            client: The HTTP client to use.
            headers: Headers for the request.
            payload: The JSON payload for the request.
            out_path: The path to save the audio file to.
            max_elapsed_seconds: Timeout for the request.

        Returns:
            The path to the saved audio file.

        Raises:
            RuntimeError: If the API returns a non-200 status code.
            TimeoutError: If the request takes longer than max_elapsed_seconds.
        """
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
        """
        Execute voice synthesis tasks via TaskRunner.

        Orchestrates the concurrent synthesis of audio files for the given worklist.

        Args:
            worklist: List of (key, text) tuples to process.
            output_dir: Directory to save audio files.
            settings: Voice generation settings.
            summary: Reporter for collecting statistics.
            skipped_count: Number of items already skipped (for reporting).

        Returns:
            A list of VoiceResult tuples (key, status).
        """
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

                async def coro(text=text, out_path=out_path):
                    """
                    Synthesize a single utterance and return the output path.

                    Returns:
                        Path to the synthesized audio file.
                    """
                    payload = await self._build_payload(text, settings)
                    return await self.synthesize_once(
                        client=client,
                        headers=headers,
                        payload=payload,
                        out_path=out_path,
                        max_elapsed_seconds=settings.max_elapsed_seconds,
                    )

                work_items.append((key, coro))

            def success_cb(spec: TaskSpec, _result: Path) -> None:
                """
                Record a successful voice synthesis.

                Args:
                    spec: TaskSpec identifying the utterance.
                    _result: Output path of the synthesized audio.
                """
                results.append((spec.task_id, "ok"))
                summary.record_success(spec.task_id)

            def failure_cb(spec: TaskSpec, exc: BaseException) -> None:
                """
                Record a failed voice synthesis.

                Args:
                    spec: TaskSpec identifying the utterance.
                    exc: Exception raised by the provider call.
                """
                logger.error(f"[VOICE] {spec.task_id} failed: {exc}")
                results.append((spec.task_id, "error"))
                summary.record_failure(spec.task_id, exc)

            def retry_cb(spec: TaskSpec, attempt: int, sleep_for: float | None) -> None:
                """
                Log retry events for voice synthesis.

                Args:
                    spec: TaskSpec identifying the utterance.
                    attempt: Current attempt number.
                    sleep_for: Seconds to wait before retrying.
                """
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

    async def _build_payload(self, text: str, settings: VoiceSettings) -> VoicePayload:
        build_async = getattr(self._provider, "build_payload_async", None)
        if callable(build_async):
            return await build_async(text, settings=settings)
        return self._provider.build_payload(text, settings=settings)
