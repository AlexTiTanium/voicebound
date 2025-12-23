import importlib.metadata
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

from commands import ai_translate, ai_voice
from core.summary_reporter import SummaryReporter
from core.task_runner import RetryConfig
from core.types import AudioFormat, TranslationSummaryStatus
from utils.command_utils import ProviderSettings

if TYPE_CHECKING:
    from apis.translation_api import TranslationResult


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[openai]
api_key = "dummy-openai"

[hume_ai]
api_key = "dummy-hume"
voice_name = "ivan"
octave_version = "2"
split_utterances = true
        """,
        encoding="utf-8",
    )
    return path


def test_cli_version_fallback(monkeypatch):
    cli_path = Path(__file__).resolve().parents[1] / "src/cli.py"

    def _raise(_name: str):
        raise importlib.metadata.PackageNotFoundError("voicebound")

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    spec = importlib.util.spec_from_file_location("cli_fallback", cli_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected module spec for cli_fallback")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.APP_VERSION == "0.0.0"


def test_translate_strings_no_provider(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ai_translate, "get_translation_provider_info", lambda _name: None)
    with pytest.raises(SystemExit):
        ai_translate.translate_strings(config_path=config_path)


def test_summarize_results_counts():
    results: list["TranslationResult"] = [
        ("a", "x", TranslationSummaryStatus.TRANSLATED),
        ("b", "x", TranslationSummaryStatus.SKIPPED),
        ("c", "x", TranslationSummaryStatus.LOADED),
        ("d", "x", TranslationSummaryStatus.IGNORED),
        ("e", "x", TranslationSummaryStatus.EMPTY),
        ("f", None, (TranslationSummaryStatus.ERROR, "boom")),
    ]
    summary = ai_translate._summarize_results(results)
    assert summary["translated"] == 1
    assert summary["skipped"] == 1
    assert summary["loaded"] == 1
    assert summary["ignored"] == 1
    assert summary["empty"] == 1
    assert summary["errors"] == ["f"]


def test_generate_voice_no_provider(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ai_voice, "get_voice_provider_info", lambda _name: None)
    with pytest.raises(SystemExit):
        ai_voice.generate_voice(config_path=config_path)


def test_generate_voice_empty_worklist(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "tmp-output"
    output_dir.mkdir()
    called = {}

    def _fake_run(*_args, **_kwargs):
        called["ran"] = True

    monkeypatch.setattr(ai_voice.anyio, "run", _fake_run)
    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=output_dir,
        allowed_regex="^keep",
    )
    assert "ran" not in called


def test_generate_voice_keyboard_interrupt(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"keep_one": "Hola"}', encoding="utf-8")
    output_dir = tmp_path / "tmp-output"
    output_dir.mkdir()

    def _raise(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ai_voice.anyio, "run", _raise)
    with pytest.raises(SystemExit) as exc:
        ai_voice.generate_voice(
            config_path=config_path,
            input_file=progress,
            output_dir=output_dir,
            allowed_regex="^keep",
        )
    assert exc.value.code == 130


class DummyVoiceProvider:
    async def send_request(self, *_args, **_kwargs):
        class Response:
            status_code = 200
            content = b"ok"
            text = "ok"

        return Response()

    def build_headers(self, _settings):
        return {"h": "v"}

    def build_payload(self, _text, *, settings):
        return {"voice": settings.voice_name}


def test_run_voice_async_builds_voice_settings(tmp_path):
    provider = DummyVoiceProvider()
    settings = ProviderSettings(
        api_key="dummy",
        model="octave",
        rpm=1000,
        concurrency=1,
        retry=RetryConfig(attempts=1, backoff_base=0.0, backoff_max=0.0, jitter=False),
    )
    summary = SummaryReporter("voice")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    results = anyio.run(
        ai_voice._run_voice_async,
        [("keep_one", "Hello")],
        output_dir,
        AudioFormat.MP3,
        "octave",
        "ivan",
        True,
        "2",
        False,
        "gpt-5-nano",
        "ru_director_v1",
        None,
        provider,
        settings,
        summary,
        0,
    )

    assert results == [("keep_one", "ok")]
