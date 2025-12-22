from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from apis.voice_api import VoicePayload, VoiceSettings
    from utils.command_utils import ProviderSettings

API_URL = "https://api.elevenlabs.io/v1/text-to-speech"


@dataclass(frozen=True)
class ElevenLabsVoiceProvider:
    """
    ElevenLabs voice provider implementation.

    Handles interaction with the ElevenLabs TTS API.

    Example:
        >>> provider = ElevenLabsVoiceProvider()
    """

    key: str = "elevenlabs"
    name: str = "elevenlabs"
    default_model: str = "eleven_multilingual_v2"
    default_rpm: int = 60
    api_url: str = API_URL

    def build_headers(self, settings: ProviderSettings) -> dict[str, str]:
        """Build headers including the API key."""
        return {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": settings.api_key,
        }

    def build_payload(self, text: str, *, settings: VoiceSettings) -> VoicePayload:
        """
        Construct the JSON payload for ElevenLabs API.

        Note: ElevenLabs expects the voice ID in the URL path. We include it
        in the payload for transport to send_request and remove it there.
        """
        return {
            "text": text,
            "model_id": settings.model,
            "voice_id": settings.voice_name,
        }

    async def send_request(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: VoicePayload,
    ) -> httpx.Response:
        """Send POST request to ElevenLabs API."""
        body: dict[str, Any] = dict(payload)
        voice_id = body.pop("voice_id", None)
        if not voice_id:
            raise ValueError("ElevenLabs payload missing voice_id.")
        url = f"{self.api_url}/{voice_id}"
        return await client.post(url, headers=headers, json=body, timeout=120)
