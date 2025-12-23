from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict, cast

from loguru import logger
from openai import AsyncOpenAI

from prompts.acting_instructions import get_acting_instruction_prompt

if TYPE_CHECKING:
    import httpx

    from apis.voice_api import VoicePayload, VoiceSettings
    from core.types import AudioFormat, VoiceProviderKey
    from utils.command_utils import ProviderSettings
else:
    import httpx

API_URL = "https://api.openai.com/v1/audio/speech"


class OpenAITTSPayload(TypedDict, total=False):
    model: str
    input: str
    voice: str
    response_format: AudioFormat
    instructions: str


@dataclass(frozen=True)
class OpenAITTSVoiceProvider:
    """
    OpenAI TTS voice provider implementation.

    Uses the OpenAI audio speech API to synthesize speech.

    Example:
        >>> provider = OpenAITTSVoiceProvider()
    """

    api_key: str
    client: AsyncOpenAI = field(init=False)

    key: VoiceProviderKey = "openai_tts"
    name: VoiceProviderKey = "openai_tts"
    default_model: str = "gpt-4o-mini-tts-2025-12-15"
    default_rpm: int = 60
    api_url: str = API_URL

    def __post_init__(self) -> None:
        object.__setattr__(self, "client", AsyncOpenAI(api_key=self.api_key))

    def build_headers(self, settings: ProviderSettings) -> dict[str, str]:
        """Build headers including the API key."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        }

    def build_payload(self, text: str, *, settings: VoiceSettings) -> VoicePayload:
        """Construct the JSON payload for OpenAI TTS."""
        payload: OpenAITTSPayload = {
            "model": settings.model,
            "input": text,
            "voice": settings.voice_name,
            "response_format": settings.audio_format,
        }
        return cast("VoicePayload", payload)

    async def build_payload_async(self, text: str, *, settings: VoiceSettings) -> VoicePayload:
        """Construct the payload, optionally generating acting instructions."""
        payload = self.build_payload(text, settings=settings)
        if settings.enabled_acting_instruction:
            instructions = await self._generate_acting_instruction(
                text,
                model=settings.acting_instruction_model,
                prompt_key=settings.acting_instruction_prompt_key,
            )
            if instructions:
                payload["instructions"] = instructions
        return payload

    async def send_request(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: VoicePayload,
    ) -> httpx.Response:
        """Send POST request to OpenAI TTS API."""
        _ = client
        _ = headers
        tts_payload = _coerce_tts_payload(payload)
        response = await cast(Any, self.client.audio.speech.create)(**tts_payload)
        content = await _read_binary_response(response)
        return httpx.Response(status_code=200, content=content)

    async def _generate_acting_instruction(self, text: str, *, model: str, prompt_key: str) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": get_acting_instruction_prompt(prompt_key),
                },
                {"role": "user", "content": text},
            ],
        )
        content = response.choices[0].message.content or ""
        instruction = content.strip().strip('"').strip("'")
        logger.debug(f"[VOICE] Acting instruction: {instruction}")
        return instruction


class _BinaryResponse:
    def __init__(self, content: bytes) -> None:
        self.status_code = 200
        self.text = ""
        self.content = content


async def _read_binary_response(response: object) -> bytes:
    if isinstance(response, bytes):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    read_fn = getattr(response, "read", None)
    if callable(read_fn):
        data = read_fn()
        if hasattr(data, "__await__"):
            return await data
        return data
    return b""


def _coerce_tts_payload(payload: "VoicePayload") -> OpenAITTSPayload:
    tts_payload: OpenAITTSPayload = {}
    for key in ("model", "input", "voice", "response_format", "instructions"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            tts_payload[key] = value
    return tts_payload
