from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from providers.hume_provider import HumeVoiceProvider
from providers.openai_provider import OpenAITranslationProvider

if TYPE_CHECKING:
    from providers.types import TranslationProvider, VoiceProvider


@dataclass(frozen=True)
class TranslationProviderInfo:
    """Metadata and factory for a translation provider."""

    key: str
    name: str
    default_model: str
    default_rpm: int
    factory: Callable[[str], TranslationProvider]


@dataclass(frozen=True)
class VoiceProviderInfo:
    """Metadata and factory for a voice provider."""

    key: str
    name: str
    default_model: str
    default_rpm: int
    factory: Callable[[], VoiceProvider]


_TRANSLATION_PROVIDERS: dict[str, TranslationProviderInfo] = {
    "openai": TranslationProviderInfo(
        key="openai",
        name="openai",
        default_model=OpenAITranslationProvider.default_model,
        default_rpm=OpenAITranslationProvider.default_rpm,
        factory=OpenAITranslationProvider,
    ),
}

_VOICE_PROVIDERS: dict[str, VoiceProviderInfo] = {
    "hume_ai": VoiceProviderInfo(
        key="hume_ai",
        name="hume_ai",
        default_model=HumeVoiceProvider.default_model,
        default_rpm=HumeVoiceProvider.default_rpm,
        factory=HumeVoiceProvider,
    ),
}

_ALIASES: dict[str, str] = {
    "open_ai": "openai",
    "openia": "openai",
    "hume": "hume_ai",
    "humeai": "hume_ai",
    "hume-ai": "hume_ai",
}


def _normalize(name: str | None) -> str:
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
    lookup = _ALIASES.get(normalized, normalized)
    return _TRANSLATION_PROVIDERS.get(lookup)


def get_voice_provider_info(name: str | None) -> VoiceProviderInfo | None:
    """
    Retrieve voice provider metadata by name or alias.

    Args:
        name: The provider name (e.g., "hume_ai", "hume").

    Returns:
        VoiceProviderInfo if found, else None.
    """
    normalized = _normalize(name)
    lookup = _ALIASES.get(normalized, normalized)
    return _VOICE_PROVIDERS.get(lookup)
