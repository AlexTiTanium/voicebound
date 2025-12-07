from pathlib import Path

import pytest

from voicebound.commands import ai_translate


class DummyEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text)


class DummyCompletionResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class DummyCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, *args, **kwargs):
        return DummyCompletionResponse(self._content)


class DummyChat:
    def __init__(self, content: str):
        self.completions = DummyCompletions(content)


class DummyOpenAI:
    def __init__(self, api_key: str, content: str = "Translated text"):
        self.chat = DummyChat(content)


@pytest.fixture(autouse=True)
def patch_tiktoken(monkeypatch):
    monkeypatch.setattr(ai_translate.tiktoken, "get_encoding", lambda *_: DummyEncoding())


def write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[openai]
api_key = "dummy-key"
model = "gpt-5-nano"

[translate]
input_file = "strings.xml"
output_file = "out/values/strings.xml"
progress_file = ".cache/progress.json"
allowed_regex = "^keep"
ignore_regex = "skip"
dry_run = false
max_workers = 2
count_tokens_enabled = true
target_language = "Spanish"
provider = "openai"
        """,
        encoding="utf-8",
    )
    return config


def write_strings(tmp_path: Path) -> Path:
    xml = tmp_path / "strings.xml"
    xml.write_text(
        """<resources>
  <string name="keep_one">Hello world</string>
  <string name="skip_me">Ignore this</string>
  <string name="other">Not matched</string>
</resources>
""",
        encoding="utf-8",
    )
    return xml


def test_translate_respects_config_and_filters(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    # Patch OpenAI client to avoid network.
    monkeypatch.setattr(ai_translate, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo"))

    output_file = tmp_path / "out/values/strings.xml"

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        output_file=output_file,
    )

    assert output_file.exists()
    text = output_file.read_text(encoding="utf-8")
    # Only "keep_one" should be translated to the dummy content; others remain untouched or skipped.
    assert "Hola mundo" in text
    assert "Ignore this" in text  # ignored by regex
    assert "Not matched" in text  # not allowed, preserved
