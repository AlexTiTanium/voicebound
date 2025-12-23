from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias


class TranslationSummaryStatus(StrEnum):
    TRANSLATED = "translated"
    SKIPPED = "skipped"
    LOADED = "loaded"
    IGNORED = "ignored"
    EMPTY = "empty"
    ERROR = "error"
    DRY_RUN = "dry-run"

class AudioFormat(StrEnum):
    MP3 = "mp3"
    WAV = "wav"

class TranslationProviderKey(StrEnum):
    OPENAI = "openai"

class VoiceProviderKey(StrEnum):
    ELEVENLABS = "elevenlabs"
    HUME_AI = "hume_ai"
    OPENAI_TTS = "openai_tts"

ProviderKey: TypeAlias = TranslationProviderKey | VoiceProviderKey

TranslationProviderAlias: TypeAlias = Literal["open_ai", "openia"]
VoiceProviderAlias: TypeAlias = Literal[
    "openai-tts",
    "openai_tts",
    "openai-voice",
    "11labs",
    "eleven-labs",
    "elevenlab",
    "hume",
    "humeai",
    "hume-ai",
]
ProviderAlias: TypeAlias = TranslationProviderAlias | VoiceProviderAlias
ProviderName: TypeAlias = ProviderKey | ProviderAlias

ConfigSection: TypeAlias = ProviderKey | Literal["retry", "translate", "voice"]

RetryConfigKey: TypeAlias = Literal["attempts", "backoff_base", "backoff_max", "jitter"]
TranslateConfigKey: TypeAlias = Literal[
    "input_file",
    "output_file",
    "progress_file",
    "allowed_regex",
    "ignore_regex",
    "dry_run",
    "stop_after",
    "count_tokens_enabled",
    "target_language",
    "provider",
]
VoiceConfigKey: TypeAlias = Literal[
    "input_file",
    "output_dir",
    "allowed_regex",
    "ignore_regex",
    "stop_after",
    "target_language",
    "provider",
    "max_elapsed_seconds",
    "dry_run",
]
ProviderConfigKey: TypeAlias = Literal[
    "api_key",
    "model",
    "rpm",
    "concurrency",
    "voice_name",
    "audio_format",
    "octave_version",
    "split_utterances",
    "enabled_acting_instruction",
    "acting_instruction_model",
    "acting_instruction_prompt_key",
    "acting_input_rate_per_1m",
    "acting_output_rate_per_1m",
    "tts_input_rate_per_1m",
    "tts_output_rate_per_1m",
    "max_chars_limit",
    "pricing_free_chars",
    "pricing_rate_per_1k",
]
ConfigKey: TypeAlias = RetryConfigKey | TranslateConfigKey | VoiceConfigKey | ProviderConfigKey

HumeVoiceProviderEnum: TypeAlias = Literal["HUME_AI"]

TranslationPromptKey: TypeAlias = Literal["literary_v1"]
ActingInstructionPromptKey: TypeAlias = Literal[
    "en_director_v1",
    "es_director_v1",
    "fr_director_v1",
    "de_director_v1",
    "pt_director_v1",
    "it_director_v1",
    "zh_director_v1",
    "uk_director_v1",
    "ru_director_v1",
]

ElevenLabsModel: TypeAlias = Literal[
    "eleven_multilingual_v2",
    "eleven_monolingual_v1",
    "eleven_turbo_v2",
    "eleven_turbo_v2_5",
]

OpenAITranslationModel: TypeAlias = Literal[
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-5-nano",
]

HumeModel: TypeAlias = Literal["octave"]

OpenAITTSModel: TypeAlias = Literal[
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts-2025-12-15",
]

OpenAITTSVoice: TypeAlias = Literal[
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]
