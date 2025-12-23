from __future__ import annotations

import re
import runpy
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, cast
from xml.etree import ElementTree as ET

import anyio
import pytest
import typer

from apis import translation_api
from apis.translation_api import (
    TranslationFilters,
    TranslationProgress,
    TranslationResult,
    TranslationSettings,
)
from commands import ai_translate
from core.summary_reporter import SummaryReporter
from core.task_runner import RetryConfig
from providers import openai_provider
from utils.command_utils import ProviderSettings

if TYPE_CHECKING:
    from core.types import TranslationProviderKey


class DummyEncoding:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


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


class DummyProvider:
    def __init__(self, content: str):
        self._content = content
        self.key: TranslationProviderKey = "openai"
        self.name: TranslationProviderKey = "openai"
        self.default_model = "dummy"
        self.default_rpm = 1

    def translate_text(self, text: str, model: str, target_language: str) -> str:
        return self._content


class DummyCtx:
    def __init__(self):
        self.obj = {}

    def ensure_object(self, typ):
        if not isinstance(self.obj, typ):
            self.obj = typ()
        return self.obj

    def get(self, key, default=None):
        return self.obj.get(key, default)


@pytest.fixture(autouse=True)
def patch_tiktoken(monkeypatch):
    monkeypatch.setattr(translation_api.tiktoken, "get_encoding", lambda *_: DummyEncoding())


def write_config(tmp_path: Path) -> Path:
    output_file = tmp_path / "tmp-output/strings.xml"
    progress_file = tmp_path / "tmp-cache/progress.json"
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[openai]
api_key = "dummy-key"
model = "gpt-5-nano"
rpm = 120
concurrency = 2

[hume_ai]
api_key = "dummy-hume"
model = "octave"
voice_name = "ivan"
octave_version = "2"
split_utterances = true
rpm = 10
concurrency = 2

[retry]
attempts = 2
backoff_base = 0.2
backoff_max = 1.0
jitter = true

[translate]
input_file = "strings.xml"
output_file = "{output_file.as_posix()}"
progress_file = "{progress_file.as_posix()}"
allowed_regex = "^keep"
ignore_regex = "skip"
dry_run = false
stop_after = 0
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
    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )

    output_file = tmp_path / "strings.out.xml"

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
    config_text = config_path.read_text(encoding="utf-8").replace(
        "dry_run = false", "dry_run = true"
    )
    config_path.write_text(config_text, encoding="utf-8")

    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )

    output_file = tmp_path / "strings.out.xml"

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
    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"keep_one": "Cached text"}', encoding="utf-8")

    call_count = {"count": 0}

    def _translate_text(self, text, model, target_language):
        call_count["count"] += 1
        return "Hola cached"

    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )
    monkeypatch.setattr(
        openai_provider.OpenAITranslationProvider, "translate_text", _translate_text
    )

    output_file = tmp_path / "strings.out.xml"

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


def test_translate_stop_after_limits_worklist(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    call_count = {"count": 0}

    def _translate_text(self, text, model, target_language):
        call_count["count"] += 1
        return "Hola"

    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )
    monkeypatch.setattr(
        openai_provider.OpenAITranslationProvider, "translate_text", _translate_text
    )

    output_file = tmp_path / "strings.out.xml"

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        output_file=output_file,
        stop_after=1,
    )

    assert call_count["count"] == 1


def test_translate_skips_runner_when_all_cached(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(
        '{"keep_one": "Cached one", "keep_two": "Cached two"}',
        encoding="utf-8",
    )

    async def _fail_translate_nodes_async(*args, **kwargs):
        raise AssertionError("translate_nodes_async should not run when all items are cached")

    def _translate_text(self, text, model, target_language):
        raise AssertionError("translate_text should not be called when all items are cached")

    monkeypatch.setattr(
        translation_api.TranslationService,
        "translate_nodes_async",
        _fail_translate_nodes_async,
    )
    monkeypatch.setattr(
        openai_provider.OpenAITranslationProvider, "translate_text", _translate_text
    )
    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )

    output_file = tmp_path / "strings.out.xml"

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        output_file=output_file,
        progress_file=progress,
    )

    text = output_file.read_text(encoding="utf-8")
    assert "Cached one" in text
    assert "Cached two" in text


def test_allowed_regex_matches_chp1_and_chp1a():
    pattern = re.compile(r"(?i)^chp1(?:[a-z])?_.*")
    assert pattern.match("chp1_17_16__a")
    assert pattern.match("chp1a_17_16__a")
    assert not pattern.match("chp10_0_1__a")


def test_translate_nodes_async_dry_run_uses_distinct_nodes(tmp_path):
    provider_settings = ProviderSettings(
        api_key="dummy",
        model="dummy",
        rpm=60,
        concurrency=2,
        retry=RetryConfig(),
    )
    service = translation_api.TranslationService(DummyProvider("Hola"), provider_settings)

    nodes: list[ET.Element] = []
    for name, text in (
        ("chp1_11_1__a", "First text"),
        ("chp1_12_1__a", "Second text"),
        ("chp1_epi_1__a", "Third text"),
    ):
        node = ET.Element("string")
        node.set("name", name)
        node.text = text
        nodes.append(node)

    filters = TranslationFilters(
        allowed_pattern=re.compile(r"^chp1"),
        ignore_pattern=re.compile(r"^$"),
    )
    settings = TranslationSettings(
        model="dummy",
        target_language="es",
        dry_run=True,
        count_tokens_enabled=False,
    )
    progress = TranslationProgress(
        done={},
        progress_file=tmp_path / "progress.json",
        progress_lock=Lock(),
    )
    summary = SummaryReporter("translate")

    async def _run() -> list[TranslationResult]:
        return await service.translate_nodes_async(
            nodes,
            filters=filters,
            progress=progress,
            settings=settings,
            summary=summary,
        )

    results = anyio.run(_run)

    names = [name for name, _, _ in results]
    assert len(names) == 3
    assert set(names) == {"chp1_11_1__a", "chp1_12_1__a", "chp1_epi_1__a"}
    assert all(isinstance(status, tuple) and status[0] == "dry-run" for _, _, status in results)


def test_prepare_nodes_skips_duplicate_keys():
    service = translation_api.TranslationService(DummyProvider("Hola"), None)
    first = ET.Element("string", {"name": "dup_key"})
    first.text = "First"
    second = ET.Element("string", {"name": "dup_key"})
    second.text = "Second"

    filters = TranslationFilters(
        allowed_pattern=re.compile(r"^dup_"),
        ignore_pattern=re.compile(r"^$"),
    )
    pre_results, translate_nodes = service.prepare_nodes(
        [first, second],
        filters=filters,
        done={},
    )

    assert pre_results == []
    assert len(translate_nodes) == 1
    assert translate_nodes[0].text == "First"


def test_translate_handles_empty_and_dry_run_branch(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)
    # Disable token counting to hit zero-token branch
    cfg_text = config_path.read_text(encoding="utf-8").replace(
        "count_tokens_enabled = true", "count_tokens_enabled = false"
    )
    cfg_text = cfg_text.replace('allowed_regex = "^chp10_"', 'allowed_regex = "^empty"')
    config_path.write_text(cfg_text, encoding="utf-8")

    # empty node matches allowed_regex to exercise dry-run and empty handling
    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )
    with pytest.raises(SystemExit):
        ai_translate.translate_strings(
            config_path=config_path,
            input_file=tmp_path / "missing.xml",
        )

    # direct function coverage
    assert translation_api.clean_text(None) is None

    # trigger warning path
    cfg_text = config_path.read_text(encoding="utf-8").replace(
        'provider = "openai"', 'provider = "other"'
    )
    config_path.write_text(cfg_text, encoding="utf-8")

    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        dry_run=True,
    )

    # call translate_text directly
    dummy_provider = DummyProvider("Hola mundo")
    assert dummy_provider.translate_text("hola", "gpt-5-nano", "Spanish") == "Hola mundo"

    # _print_dry_run with payload
    ai_translate._print_dry_run(cast(list[TranslationResult], [("a", None, ("dry-run", 1, "p"))]))

    # empty branch
    node = ET.Element("string", {"name": "empty"})
    node.text = ""
    result = translation_api.process_string(
        node,
        translate_pattern=re.compile(r"^empty"),
        ignore_pattern=re.compile(r"^$"),
        done={},
        progress_lock=Lock(),
        provider=DummyProvider("Hola"),
        progress_file=tmp_path / "p.json",
        model="m",
        dry_run=False,
        encoding=None,
        target_language="es",
    )
    assert result[2] == "empty"

    # dry-run branch
    node.text = "Text"
    result = translation_api.process_string(
        node,
        translate_pattern=re.compile(r"^empty"),
        ignore_pattern=re.compile(r"^$"),
        done={},
        progress_lock=Lock(),
        provider=DummyProvider("Hola"),
        progress_file=tmp_path / "p2.json",
        model="m",
        dry_run=True,
        encoding=None,
        target_language="es",
    )
    assert isinstance(result[2], tuple) and result[2][0] == "dry-run"

    # zero-token branch (no encoding) with translation and cache update
    progress_file = tmp_path / "p3.json"
    name, translated, status = translation_api.process_string(
        node,
        translate_pattern=re.compile(r"^empty"),
        ignore_pattern=re.compile(r"^$"),
        done={},
        progress_lock=Lock(),
        provider=DummyProvider("Hola"),
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

    monkeypatch.setattr(
        openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola mundo")
    )

    async def _boom(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(translation_api.TranslationService, "translate_nodes_async", _boom)

    with pytest.raises(SystemExit):
        ai_translate.translate_strings(
            config_path=config_path,
            input_file=tmp_path / "strings.xml",
            output_file=tmp_path / "strings.out.xml",
            progress_file=tmp_path / "progress.json",
        )


def test_translate_typer_command(monkeypatch, tmp_path):
    called = {}

    def fake_translate_strings(**kwargs):
        called["ok"] = kwargs

    monkeypatch.setattr(ai_translate, "translate_strings", fake_translate_strings)

    ai_translate.typer_command(
        cast(typer.Context, DummyCtx()),
        input_file=Path("a.xml"),
        output_file=Path("b.xml"),
        allowed_regex="^x",
        ignore_regex="y",
        dry_run=True,
        model="m",
        target_language="es",
        config_path=Path("config.toml"),
    )

    assert called["ok"]["input_file"] == Path("a.xml")


def test_ai_translate_main_entry(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["voicebound-translate", "--help"])
    try:
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "src/commands/ai_translate.py"),
            run_name="__main__",
        )
    except SystemExit as exc:
        assert exc.code == 0


def test_translate_reports_errors(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_strings(tmp_path)

    monkeypatch.setattr(openai_provider, "OpenAI", lambda api_key: DummyOpenAI(api_key, "Hola"))

    async def _return_error(self, *args, **kwargs):
        return [("bad", None, ("error", "boom"))]

    monkeypatch.setattr(translation_api.TranslationService, "translate_nodes_async", _return_error)

    output_file = tmp_path / "strings.out.xml"
    ai_translate.translate_strings(
        config_path=config_path,
        input_file=tmp_path / "strings.xml",
        output_file=output_file,
    )
    assert output_file.exists()
    summary = ai_translate._summarize_results(
        cast(list[TranslationResult], [("bad", None, ("error", "boom"))])
    )
    assert summary["errors"] == ["bad"]
