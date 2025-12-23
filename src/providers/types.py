from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import httpx

    from apis.voice_api import VoicePayload, VoiceSettings
    from core.types import TranslationProviderKey, VoiceProviderKey
    from utils.command_utils import ProviderSettings


class TranslationProvider(Protocol):
    """
    Protocol defining the interface for translation providers.

    Example:
        >>> provider.translate_text("Hello", "gpt-5-nano", "Spanish")
        'Hola'
    """

    key: TranslationProviderKey
    name: TranslationProviderKey
    default_model: str
    default_rpm: int

    def translate_text(self, text: str, model: str, target_language: str) -> str:
        """
        Translate text from source to target language.

        Args:
            text: The text to translate.
            model: The model identifier.
            target_language: The target language.

        Returns:
            The translated text.
        """
        ...


class VoiceProvider(Protocol):
    """
    Protocol defining the interface for voice synthesis providers.

    Example:
        >>> headers = provider.build_headers(settings)
    """

    key: VoiceProviderKey
    name: VoiceProviderKey
    default_model: str
    default_rpm: int

    def build_headers(self, settings: ProviderSettings) -> dict[str, str]:
        """
        Build HTTP headers for the API request.

        Args:
            settings: Provider configuration settings.

        Returns:
            A dictionary of headers.
        """
        ...

    def build_payload(self, text: str, *, settings: VoiceSettings) -> VoicePayload:
        """
        Build the JSON payload for the API request.

        Args:
            text: The text to synthesize.
            settings: Voice generation settings.

        Returns:
            A VoicePayload dictionary.
        """
        ...

    async def send_request(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: VoicePayload,
    ) -> httpx.Response:
        """
        Send the synthesis request to the provider API.

        Args:
            client: The HTTP client.
            headers: Request headers.
            payload: Request payload.

        Returns:
            The HTTP response.
        """
        ...
