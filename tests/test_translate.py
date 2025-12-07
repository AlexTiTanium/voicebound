import runpy
import re
from pathlib import Path
from threading import Lock
from xml.etree import ElementTree as ET

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
  <string name="keep_two">Hello again</string>
  <string name="skip_me">Ignore this</string>
  <string name="other">Not matched</string>
  <string name="empty"></string>
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
    assert "Hola mundo" in text


def test_translate_dry_run_does_not_write(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    # Force dry run via config override
    config_text = (
        config_path.read_text(encoding="utf-8").replace("dry_run = false", "dry_run = true")
    )
    config_path.write_text(config_text, encoding="utf-8")

    monkeypatch.setattr(ai_translate, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo"))

    output_file = tmp_path / "out/values/strings.xml"

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        output_file=output_file,
    )

    assert not output_file.exists()
    # Ensure dry run summary path is hit by invoking with empty results
    ai_translate._print_dry_run([])


def test_translate_uses_cache_and_skips_api(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    # Prepare cache with pretranslated value
    progress = tmp_path / ".cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"keep_one": "Cached text"}', encoding="utf-8")

    call_count = {"count": 0}

    def _translate_text(client, text, model, target_language):
        call_count["count"] += 1
        return "Hola cached"

    monkeypatch.setattr(ai_translate, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo"))
    monkeypatch.setattr(ai_translate, "translate_text", _translate_text)

    output_file = tmp_path / "out/values/strings.xml"

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        output_file=output_file,
        progress_file=progress,
    )

    text = output_file.read_text(encoding="utf-8")
    assert "Cached text" in text
    # Only one uncached item should invoke translation
    assert call_count["count"] == 1


def test_translate_handles_empty_and_dry_run_branch(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)
    # Disable token counting to hit zero-token branch
    cfg_text = config_path.read_text(encoding="utf-8").replace("count_tokens_enabled = true", "count_tokens_enabled = false")
    cfg_text = cfg_text.replace('allowed_regex = "^chp10_"', 'allowed_regex = "^empty"')
    config_path.write_text(cfg_text, encoding="utf-8")

    # empty node matches allowed_regex to exercise dry-run and empty handling
    monkeypatch.setattr(ai_translate, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo"))
    with pytest.raises(SystemExit):
        ai_translate.translate_strings(
            config_path=config_path,
            input_file=tmp_path / "missing.xml",
        )

    # direct function coverage
    assert ai_translate.clean_text(None) is None

    # trigger warning path
    cfg_text = config_path.read_text(encoding="utf-8").replace('provider = "openai"', 'provider = "other"')
    config_path.write_text(cfg_text, encoding="utf-8")

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        dry_run=True,
    )

    # call translate_text directly
    dummy_client = DummyOpenAI("k", "Hola mundo")
    assert ai_translate.translate_text(dummy_client, "hola", "gpt-5-nano", "Spanish") == "Hola mundo"

    # _print_dry_run with payload
    ai_translate._print_dry_run([("a", None, ("dry-run", 1, "p"))])

    # empty branch
    node = ET.Element("string", {"name": "empty"})
    node.text = ""
    result = ai_translate.process_string(
        node,
        translate_pattern=re.compile(r"^empty"),
        ignore_pattern=re.compile(r"^$"),
        done={},
        progress_lock=Lock(),
        client=DummyOpenAI("k", "Hola"),
        progress_file=tmp_path / "p.json",
        model="m",
        dry_run=False,
        encoding=None,
        target_language="es",
    )
    assert result[2] == "empty"

    # dry-run branch
    node.text = "Text"
    result = ai_translate.process_string(
        node,
        translate_pattern=re.compile(r"^empty"),
        ignore_pattern=re.compile(r"^$"),
        done={},
        progress_lock=Lock(),
        client=DummyOpenAI("k", "Hola"),
        progress_file=tmp_path / "p2.json",
        model="m",
        dry_run=True,
        encoding=None,
        target_language="es",
    )
    assert isinstance(result[2], tuple) and result[2][0] == "dry-run"

    # zero-token branch (no encoding) with translation and cache update
    progress_file = tmp_path / "p3.json"
    name, translated, status = ai_translate.process_string(
        node,
        translate_pattern=re.compile(r"^empty"),
        ignore_pattern=re.compile(r"^$"),
        done={},
        progress_lock=Lock(),
        client=DummyOpenAI("k", "Hola"),
        progress_file=progress_file,
        model="m",
        dry_run=False,
        encoding=None,
        target_language="es",
    )
    assert status == "translated"
    assert translated == "Hola"


def test_translate_keyboard_interrupt(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    monkeypatch.setattr(ai_translate, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo"))
    def _boom(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(ai_translate, "as_completed", _boom)

    with pytest.raises(SystemExit):
        ai_translate.translate_strings(
            config_path=config_path,
            input_file=tmp_path / "strings.xml",
            output_file=tmp_path / "out/values/strings.xml",
        )


def test_translate_typer_command(monkeypatch, tmp_path):
    called = {}

    def fake_translate_strings(**kwargs):
        called["ok"] = kwargs

    monkeypatch.setattr(ai_translate, "translate_strings", fake_translate_strings)

    ai_translate.typer_command(
        input_file=Path("a.xml"),
        output_file=Path("b.xml"),
        allowed_regex="^x",
        ignore_regex="y",
        dry_run=True,
        max_workers=1,
        model="m",
        target_language="es",
        config_path=Path("config.toml"),
    )

    assert called["ok"]["input_file"] == Path("a.xml")


def test_ai_translate_main_entry(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["voicebound-translate", "--help"])
    try:
        runpy.run_path(
            Path(__file__).resolve().parents[1] / "src/voicebound/commands/ai_translate.py",
            run_name="__main__",
        )
    except SystemExit as exc:
        assert exc.code == 0
