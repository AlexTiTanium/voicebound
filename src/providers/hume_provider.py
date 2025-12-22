from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from apis.voice_api import VoicePayload, VoiceSettings
    from utils.command_utils import ProviderSettings

API_URL = "https://api.hume.ai/v0/tts/file"
VOICE_PROVIDER_ENUM = "HUME_AI"


@dataclass(frozen=True)
class HumeVoiceProvider:
    """
    Hume AI voice provider implementation.

    Handles interaction with the Hume AI TTS API.

    Example:
        >>> provider = HumeVoiceProvider()
    """

    key: str = "hume_ai"
    name: str = "hume_ai"
    default_model: str = "octave"
    default_rpm: int = 10
    api_url: str = API_URL

    def build_headers(self, settings: ProviderSettings) -> dict[str, str]:
        """Build headers including the API key."""
        return {
            "Content-Type": "application/json",
            "X-Hume-Api-Key": settings.api_key,
        }

    def build_payload(self, text: str, *, settings: VoiceSettings) -> VoicePayload:
        """Construct the JSON payload for Hume API."""
        return {
            "model": settings.model,
            "format": {"type": settings.audio_format},
            "split_utterances": settings.split_utterances,
            "version": settings.octave_version,
            "utterances": [
                {
                    "text": text,
                    "voice": {"name": settings.voice_name, "provider": VOICE_PROVIDER_ENUM},
                }
            ],
        }

    async def send_request(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: VoicePayload,
    ) -> httpx.Response:
        """Send POST request to Hume API."""
        return await client.post(self.api_url, headers=headers, json=payload, timeout=120)
