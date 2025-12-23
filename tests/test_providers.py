from typing import Any, cast

import httpx
import pytest

from apis.voice_api import VoicePayload, VoiceSettings
from core.task_runner import RetryConfig
from core.types import AudioFormat, TranslationProviderKey, VoiceProviderKey
from prompts.acting_instructions import (
    DEFAULT_ACTING_INSTRUCTION_PROMPT_KEY,
    get_acting_instruction_prompt,
)
from prompts.translation import (
    DEFAULT_TRANSLATION_PROMPT_KEY,
    get_translation_prompt,
)
from providers import (
    elevenlabs_provider,
    hume_provider,
    openai_provider,
    openai_tts_provider,
    registry,
)
from utils.command_utils import ProviderSettings


class DummyCompletionResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class DummyCompletions:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return DummyCompletionResponse(self._content)


class DummyChat:
    def __init__(self, content: str):
        self.completions = DummyCompletions(content)


class DummyOpenAI:
    def __init__(self, api_key: str, content: str = "  Hola \n"):
        self.api_key = api_key
        self.chat = DummyChat(content)


class DummyAsyncCompletions:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return DummyCompletionResponse(self._content)


class DummyAsyncChat:
    def __init__(self, content: str):
        self.completions = DummyAsyncCompletions(content)


class DummyAudioSpeech:
    def __init__(self, content: bytes):
        self._content = content
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._content


class DummyAudio:
    def __init__(self, content: bytes):
        self.speech = DummyAudioSpeech(content)


class DummyAsyncOpenAI:
    def __init__(self, api_key: str, content: str = "Act softly.", audio: bytes = b"audio"):
        self.api_key = api_key
        self.chat = DummyAsyncChat(content)
        self.audio = DummyAudio(audio)


def test_openai_translation_provider_builds_prompt(monkeypatch):
    monkeypatch.setattr(openai_provider, "OpenAI", DummyOpenAI)
    provider = openai_provider.OpenAITranslationProvider(api_key="sk-test")

    result = provider.translate_text("Hello world", "gpt-5-nano", "Spanish")

    assert result == "Hola"
    client = cast(DummyOpenAI, provider.client)
    call = cast(dict[str, Any], client.chat.completions.calls[0])
    assert call["model"] == "gpt-5-nano"
    messages = cast(list[dict[str, Any]], call["messages"])
    message = cast(str, messages[0]["content"])
    assert "Spanish" in message
    assert "Hello world" in message


def test_hume_provider_builds_headers_and_payload():
    provider = hume_provider.HumeVoiceProvider()
    settings = ProviderSettings(
        api_key="hume-key",
        model="octave",
        rpm=10,
        concurrency=1,
        retry=RetryConfig(),
    )
    voice_settings = VoiceSettings(
        model="octave",
        voice_name="ivan",
        audio_format=AudioFormat.MP3,
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )

    headers = provider.build_headers(settings)
    payload = provider.build_payload("hello", settings=voice_settings)

    assert headers["X-Hume-Api-Key"] == "hume-key"
    assert payload["model"] == "octave"
    assert payload["format"]["type"] == AudioFormat.MP3
    assert payload["split_utterances"] is True
    assert payload["version"] == "2"
    assert payload["utterances"][0]["text"] == "hello"
    assert payload["utterances"][0]["voice"]["name"] == "ivan"
    assert payload["utterances"][0]["voice"]["provider"] == "HUME_AI"


def test_hume_provider_payload_schema():
    provider = hume_provider.HumeVoiceProvider()
    voice_settings = VoiceSettings(
        model="octave",
        voice_name="ivan",
        audio_format=AudioFormat.MP3,
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )

    payload = provider.build_payload("Hello", settings=voice_settings)

    assert set(payload.keys()) == {"model", "format", "split_utterances", "version", "utterances"}
    assert payload["format"] == {"type": AudioFormat.MP3}
    assert isinstance(payload["utterances"], list)
    assert len(payload["utterances"]) == 1

    utterance = payload["utterances"][0]
    assert set(utterance.keys()) == {"text", "voice"}
    assert utterance["text"] == "Hello"
    assert set(utterance["voice"].keys()) == {"name", "provider"}
    assert utterance["voice"]["name"] == "ivan"
    assert utterance["voice"]["provider"] == hume_provider.VOICE_PROVIDER_ENUM


def test_hume_provider_send_request_uses_client():
    provider = hume_provider.HumeVoiceProvider()
    calls = {}

    class DummyResponse:
        pass

    class DummyClient:
        async def post(self, url, headers=None, json=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            calls["json"] = json
            calls["timeout"] = timeout
            return DummyResponse()

    payload: VoicePayload = {
        "model": "octave",
        "format": {"type": AudioFormat.MP3},
        "split_utterances": True,
        "version": "2",
        "utterances": [{"text": "hello", "voice": {"name": "ivan", "provider": "HUME_AI"}}],
    }

    import asyncio

    response = asyncio.run(
        provider.send_request(
            client=cast(httpx.AsyncClient, DummyClient()),
            headers={"h": "v"},
            payload=payload,
        )
    )

    assert isinstance(response, DummyResponse)
    assert calls["url"] == hume_provider.API_URL
    assert calls["headers"] == {"h": "v"}
    assert calls["json"] == payload
    assert calls["timeout"] == 120


def test_elevenlabs_provider_builds_headers_and_payload():
    provider = elevenlabs_provider.ElevenLabsVoiceProvider()
    settings = ProviderSettings(
        api_key="eleven-key",
        model="eleven_multilingual_v2",
        rpm=60,
        concurrency=1,
        retry=RetryConfig(),
    )
    voice_settings = VoiceSettings(
        model="eleven_multilingual_v2",
        voice_name="voice-id",
        audio_format=AudioFormat.MP3,
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )

    headers = provider.build_headers(settings)
    payload = provider.build_payload("hello", settings=voice_settings)

    assert headers["xi-api-key"] == "eleven-key"
    assert headers["Accept"] == "audio/mpeg"
    assert payload["text"] == "hello"
    assert payload["model_id"] == "eleven_multilingual_v2"
    assert payload["voice_id"] == "voice-id"


def test_elevenlabs_provider_send_request_uses_voice_id():
    provider = elevenlabs_provider.ElevenLabsVoiceProvider()
    calls = {}

    class DummyResponse:
        pass

    class DummyClient:
        async def post(self, url, headers=None, json=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            calls["json"] = json
            calls["timeout"] = timeout
            return DummyResponse()

    payload = {
        "text": "hello",
        "model_id": "eleven_multilingual_v2",
        "voice_id": "voice-id",
    }

    import asyncio

    response = asyncio.run(
        provider.send_request(
            client=cast(httpx.AsyncClient, DummyClient()),
            headers={"h": "v"},
            payload=cast(VoicePayload, payload),
        )
    )

    assert isinstance(response, DummyResponse)
    assert calls["url"] == f"{elevenlabs_provider.API_URL}/voice-id"
    assert calls["headers"] == {"h": "v"}
    assert calls["json"] == {"text": "hello", "model_id": "eleven_multilingual_v2"}
    assert calls["timeout"] == 120


def test_elevenlabs_provider_requires_voice_id():
    provider = elevenlabs_provider.ElevenLabsVoiceProvider()

    class DummyClient:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("Client should not be called without voice_id.")

    payload = {"text": "hello", "model_id": "eleven_multilingual_v2"}

    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(
            provider.send_request(
                client=cast(httpx.AsyncClient, DummyClient()),
                headers={"h": "v"},
                payload=cast(VoicePayload, payload),
            )
        )


def test_openai_tts_provider_builds_headers_and_payload(monkeypatch):
    dummy_client = DummyAsyncOpenAI("openai-key")
    monkeypatch.setattr(
        openai_tts_provider,
        "AsyncOpenAI",
        lambda api_key: dummy_client,
    )
    provider = openai_tts_provider.OpenAITTSVoiceProvider(api_key="openai-key")
    settings = ProviderSettings(
        api_key="openai-key",
        model="gpt-4o-mini-tts-2025-12-15",
        rpm=60,
        concurrency=1,
        retry=RetryConfig(),
    )
    voice_settings = VoiceSettings(
        model="gpt-4o-mini-tts-2025-12-15",
        voice_name="onyx",
        audio_format=AudioFormat.MP3,
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
        enabled_acting_instruction=False,
        acting_instruction_model="gpt-5-nano",
    )

    headers = provider.build_headers(settings)
    import asyncio

    payload = asyncio.run(provider.build_payload_async("hello", settings=voice_settings))

    assert headers["Authorization"] == "Bearer openai-key"
    assert payload["model"] == "gpt-4o-mini-tts-2025-12-15"
    assert payload["input"] == "hello"
    assert payload["voice"] == "onyx"
    assert payload["response_format"] == AudioFormat.MP3
    assert "instructions" not in payload


def test_openai_tts_provider_generates_instructions_when_enabled(monkeypatch):
    dummy_client = DummyAsyncOpenAI("openai-key", content="Narrate with urgency.")
    monkeypatch.setattr(
        openai_tts_provider,
        "AsyncOpenAI",
        lambda api_key: dummy_client,
    )
    provider = openai_tts_provider.OpenAITTSVoiceProvider(api_key="openai-key")
    voice_settings = VoiceSettings(
        model="gpt-4o-mini-tts-2025-12-15",
        voice_name="onyx",
        audio_format=AudioFormat.MP3,
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
        enabled_acting_instruction=True,
        acting_instruction_model="gpt-5-nano",
    )

    import asyncio

    payload = asyncio.run(provider.build_payload_async("Hello", settings=voice_settings))

    assert payload["instructions"] == "Narrate with urgency."
    call = dummy_client.chat.completions.calls[0]
    assert call["model"] == "gpt-5-nano"
    messages = cast(list[dict[str, Any]], call["messages"])
    assert messages[1]["content"] == "Hello"


def test_openai_tts_provider_send_request_uses_client(monkeypatch):
    dummy_client = DummyAsyncOpenAI("openai-key", audio=b"audio-data")
    monkeypatch.setattr(
        openai_tts_provider,
        "AsyncOpenAI",
        lambda api_key: dummy_client,
    )
    provider = openai_tts_provider.OpenAITTSVoiceProvider(api_key="openai-key")

    payload: VoicePayload = {
        "model": "gpt-4o-mini-tts-2025-12-15",
        "input": "hello",
        "voice": "onyx",
        "response_format": AudioFormat.MP3,
        "instructions": "Test",
    }

    import asyncio

    response = asyncio.run(
        provider.send_request(
            client=cast(httpx.AsyncClient, object()),
            headers={"h": "v"},
            payload=payload,
        )
    )

    assert response.status_code == 200
    assert response.content == b"audio-data"
    call = dummy_client.audio.speech.calls[0]
    assert call == payload


def test_openai_tts_read_binary_response_variants():
    async def _run():
        assert await openai_tts_provider._read_binary_response(b"raw") == b"raw"

        class WithContent:
            content = b"bytes"

        assert await openai_tts_provider._read_binary_response(WithContent()) == b"bytes"

        class WithAsyncRead:
            async def read(self):
                return b"async"

        assert await openai_tts_provider._read_binary_response(WithAsyncRead()) == b"async"

        class WithSyncRead:
            def read(self):
                return b"sync"

        assert await openai_tts_provider._read_binary_response(WithSyncRead()) == b"sync"

        assert await openai_tts_provider._read_binary_response(object()) == b""

    import asyncio

    asyncio.run(_run())


def test_openai_tts_binary_response_wrapper():
    wrapper = openai_tts_provider._BinaryResponse(b"data")
    assert wrapper.status_code == 200
    assert wrapper.text == ""
    assert wrapper.content == b"data"


def test_acting_instruction_prompt_registry():
    prompt = get_acting_instruction_prompt(DEFAULT_ACTING_INSTRUCTION_PROMPT_KEY)
    assert "voice director" in prompt.lower()

    with pytest.raises(ValueError):
        get_acting_instruction_prompt("missing-key")


def test_translation_prompt_registry():
    prompt = get_translation_prompt(
        DEFAULT_TRANSLATION_PROMPT_KEY,
        target_language="Spanish",
        text="Hello",
    )
    assert "Spanish" in prompt
    assert "Hello" in prompt

    with pytest.raises(ValueError):
        get_translation_prompt("missing-key", target_language="Spanish", text="Hello")


def test_registry_resolves_aliases_and_missing():
    info = registry.get_translation_provider_info("Open_AI")
    assert info is not None
    assert info.key == TranslationProviderKey.OPENAI

    voice_info = registry.get_voice_provider_info("hume-ai")
    assert voice_info is not None
    assert voice_info.key == VoiceProviderKey.HUME_AI

    voice_info = registry.get_voice_provider_info("11labs")
    assert voice_info is not None
    assert voice_info.key == VoiceProviderKey.ELEVENLABS

    voice_info = registry.get_voice_provider_info("openai-tts")
    assert voice_info is not None
    assert voice_info.key == VoiceProviderKey.OPENAI_TTS

    assert registry.get_translation_provider_info(None) is None
