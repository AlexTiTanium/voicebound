from __future__ import annotations

from typing import Literal, TypeAlias

TranslationProviderKey: TypeAlias = Literal["openai"]
VoiceProviderKey: TypeAlias = Literal["elevenlabs", "hume_ai", "openai_tts"]
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

AudioFormat: TypeAlias = Literal["mp3", "wav"]
HumeVoiceProviderEnum: TypeAlias = Literal["HUME_AI"]

TranslationSummaryStatus: TypeAlias = Literal[
    "translated",
    "skipped",
    "loaded",
    "ignored",
    "empty",
    "error",
    "dry-run",
]
