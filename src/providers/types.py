from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import httpx

    from apis.voice_api import VoicePayload, VoiceSettings
    from utils.command_utils import ProviderSettings


class TranslationProvider(Protocol):
    key: str
    name: str
    default_model: str
    default_rpm: int

    def translate_text(self, text: str, model: str, target_language: str) -> str: ...


class VoiceProvider(Protocol):
    key: str
    name: str
    default_model: str
    default_rpm: int

    def build_headers(self, settings: ProviderSettings) -> dict[str, str]: ...

    def build_payload(self, text: str, *, settings: VoiceSettings) -> VoicePayload: ...

    async def send_request(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: VoicePayload,
    ) -> httpx.Response: ...
