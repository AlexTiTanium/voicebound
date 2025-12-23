from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

from core.types import ProviderAlias, ProviderKey, TranslationProviderKey, VoiceProviderKey
from providers.elevenlabs_provider import ElevenLabsVoiceProvider
from providers.hume_provider import HumeVoiceProvider
from providers.openai_provider import OpenAITranslationProvider
from providers.openai_tts_provider import OpenAITTSVoiceProvider

if TYPE_CHECKING:
    from providers.types import TranslationProvider, VoiceProvider


@dataclass(frozen=True)
class TranslationProviderInfo:
    """
    Metadata and factory for a translation provider.

    Example:
        >>> info = get_translation_provider_info("openai")
    """

    key: TranslationProviderKey
    name: TranslationProviderKey
    default_model: str
    default_rpm: int
    factory: Callable[[str], TranslationProvider]


@dataclass(frozen=True)
class VoiceProviderInfo:
    """
    Metadata and factory for a voice provider.

    Example:
        >>> info = get_voice_provider_info("hume_ai")
    """

    key: VoiceProviderKey
    name: VoiceProviderKey
    default_model: str
    default_voice_name: str
    default_rpm: int
    factory: Callable[[str], VoiceProvider]


_TRANSLATION_PROVIDERS: dict[TranslationProviderKey, TranslationProviderInfo] = {
    TranslationProviderKey.OPENAI: TranslationProviderInfo(
        key=TranslationProviderKey.OPENAI,
        name=TranslationProviderKey.OPENAI,
        default_model=OpenAITranslationProvider.default_model,
        default_rpm=OpenAITranslationProvider.default_rpm,
        factory=OpenAITranslationProvider,
    ),
}

_VOICE_PROVIDERS: dict[VoiceProviderKey, VoiceProviderInfo] = {
    VoiceProviderKey.ELEVENLABS: VoiceProviderInfo(
        key=VoiceProviderKey.ELEVENLABS,
        name=VoiceProviderKey.ELEVENLABS,
        default_model=ElevenLabsVoiceProvider.default_model,
        default_voice_name="VOICE_ID",
        default_rpm=ElevenLabsVoiceProvider.default_rpm,
        factory=lambda _api_key: ElevenLabsVoiceProvider(),
    ),
    VoiceProviderKey.HUME_AI: VoiceProviderInfo(
        key=VoiceProviderKey.HUME_AI,
        name=VoiceProviderKey.HUME_AI,
        default_model=HumeVoiceProvider.default_model,
        default_voice_name="ivan",
        default_rpm=HumeVoiceProvider.default_rpm,
        factory=lambda _api_key: HumeVoiceProvider(),
    ),
    VoiceProviderKey.OPENAI_TTS: VoiceProviderInfo(
        key=VoiceProviderKey.OPENAI_TTS,
        name=VoiceProviderKey.OPENAI_TTS,
        default_model=OpenAITTSVoiceProvider.default_model,
        default_voice_name="onyx",
        default_rpm=OpenAITTSVoiceProvider.default_rpm,
        factory=OpenAITTSVoiceProvider,
    ),
}

_ALIASES: dict[ProviderAlias, ProviderKey] = {
    "open_ai": TranslationProviderKey.OPENAI,
    "openia": TranslationProviderKey.OPENAI,
    "openai-tts": VoiceProviderKey.OPENAI_TTS,
    "openai_tts": VoiceProviderKey.OPENAI_TTS,
    "openai-voice": VoiceProviderKey.OPENAI_TTS,
    "11labs": VoiceProviderKey.ELEVENLABS,
    "eleven-labs": VoiceProviderKey.ELEVENLABS,
    "elevenlab": VoiceProviderKey.ELEVENLABS,
    "hume": VoiceProviderKey.HUME_AI,
    "humeai": VoiceProviderKey.HUME_AI,
    "hume-ai": VoiceProviderKey.HUME_AI,
}


def _normalize(name: str | None) -> str:
    """
    Normalize provider names for lookup.

    Args:
        name: User-supplied provider name or alias.

    Returns:
        Normalized lowercase name or empty string.
    """
    if not name:
        return ""
    return name.strip().lower()


def get_translation_provider_info(name: str | None) -> TranslationProviderInfo | None:
    """
    Retrieve translation provider metadata by name or alias.

    Args:
        name: The provider name (e.g., "openai", "open_ai").

    Returns:
        TranslationProviderInfo if found, else None.
    """
    normalized = _normalize(name)
    if normalized in _ALIASES:
        lookup = _ALIASES[cast(ProviderAlias, normalized)]
    else:
        lookup = normalized
    if lookup in _TRANSLATION_PROVIDERS:
        return _TRANSLATION_PROVIDERS[cast(TranslationProviderKey, lookup)]
    return None


def get_voice_provider_info(name: str | None) -> VoiceProviderInfo | None:
    """
    Retrieve voice provider metadata by name or alias.

    Args:
        name: The provider name (e.g., "hume_ai", "hume").

    Returns:
        VoiceProviderInfo if found, else None.
    """
    normalized = _normalize(name)
    if normalized in _ALIASES:
        lookup = _ALIASES[cast(ProviderAlias, normalized)]
    else:
        lookup = normalized
    if lookup in _VOICE_PROVIDERS:
        return _VOICE_PROVIDERS[cast(VoiceProviderKey, lookup)]
    return None
