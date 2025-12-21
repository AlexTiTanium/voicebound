from apis.voice_api import VoiceSettings
from core.task_runner import RetryConfig
from providers import hume_provider, openai_provider, registry
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


def test_openai_translation_provider_builds_prompt(monkeypatch):
    monkeypatch.setattr(openai_provider, "OpenAI", DummyOpenAI)
    provider = openai_provider.OpenAITranslationProvider(api_key="sk-test")

    result = provider.translate_text("Hello world", "gpt-5-nano", "Spanish")

    assert result == "Hola"
    call = provider.client.chat.completions.calls[0]
    assert call["model"] == "gpt-5-nano"
    message = call["messages"][0]["content"]
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
        provider="hume_ai",
        audio_format="mp3",
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )

    headers = provider.build_headers(settings)
    payload = provider.build_payload("hello", settings=voice_settings)

    assert headers["X-Hume-Api-Key"] == "hume-key"
    assert payload["model"] == "octave"
    assert payload["format"]["type"] == "mp3"
    assert payload["split_utterances"] is True
    assert payload["version"] == "2"
    assert payload["utterances"][0]["text"] == "hello"
    assert payload["utterances"][0]["voice"]["name"] == "ivan"
    assert payload["utterances"][0]["voice"]["provider"] == "hume_ai"


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

    import asyncio

    response = asyncio.run(
        provider.send_request(
            client=DummyClient(),
            headers={"h": "v"},
            payload={"ok": True},
        )
    )

    assert isinstance(response, DummyResponse)
    assert calls["url"] == hume_provider.API_URL
    assert calls["headers"] == {"h": "v"}
    assert calls["json"] == {"ok": True}
    assert calls["timeout"] == 120


def test_registry_resolves_aliases_and_missing():
    info = registry.get_translation_provider_info("Open_AI")
    assert info is not None
    assert info.key == "openai"

    voice_info = registry.get_voice_provider_info("hume-ai")
    assert voice_info is not None
    assert voice_info.key == "hume_ai"

    assert registry.get_translation_provider_info(None) is None
