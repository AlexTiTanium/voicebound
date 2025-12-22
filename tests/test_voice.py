import runpy
from pathlib import Path
from typing import cast

import pytest
import typer

from commands import ai_voice


class DummyCtx:
    def __init__(self):
        self.obj = {}

    def ensure_object(self, typ):
        if not isinstance(self.obj, typ):
            self.obj = typ()
        return self.obj

    def get(self, key, default=None):
        return self.obj.get(key, default)


def write_config(tmp_path: Path) -> Path:
    input_file = tmp_path / "tmp-cache/progress.json"
    output_dir = tmp_path / "tmp-output/hume"
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[openai]
api_key = "dummy-openai"
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

[voice]
input_file = "{input_file.as_posix()}"
output_dir = "{output_dir.as_posix()}"
audio_format = "mp3"
allowed_regex = "^keep"
ignore_regex = "skip"
stop_after = 0
target_language = "Spanish"
provider = "hume_ai"
jitter_fraction = 0.1
max_elapsed_seconds = 5

[retry]
attempts = 2
backoff_base = 0.2
backoff_max = 1.0
jitter = true
        """,
        encoding="utf-8",
    )
    return config


def write_progress(tmp_path: Path) -> Path:
    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(
        '{"keep_one": "Hola mundo", "skip_me": "ignored", "other": "nope", "keep_two": "Hola"}',
        encoding="utf-8",
    )
    return progress


@pytest.fixture()
def stub_runner(monkeypatch):
    calls = {}

    async def _fake_run(worklist, *_args, **_kwargs):
        calls["worklist"] = worklist
        return [(key, "ok") for key, _ in worklist]

    monkeypatch.setattr(ai_voice, "_run_voice_async", _fake_run)
    return calls


def test_voice_filters_and_invokes_runner(monkeypatch, tmp_path, stub_runner):
    config_path = write_config(tmp_path)
    progress = write_progress(tmp_path)

    out_dir = tmp_path / "tmp-output/hume"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pretend keep_one already exists
    (out_dir / "keep_one.mp3").write_bytes(b"existing")

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
        ignore_regex="skip",
    )

    worklist = stub_runner["worklist"]
    keys = [item[0] for item in worklist]
    assert "keep_one" not in keys  # skipped because file exists
    assert "keep_two" in keys
    assert "skip_me" not in keys
    assert "other" not in keys


def test_voice_dry_run_skips_runner(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    progress = write_progress(tmp_path)
    called = {}

    async def _fail(*_args, **_kwargs):
        called["ran"] = True

    monkeypatch.setattr(ai_voice, "_run_voice_async", _fail)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=tmp_path / "tmp-output/hume",
        allowed_regex="^keep",
        dry_run=True,
    )

    assert "ran" not in called


def test_voice_dry_run_elevenlabs_warns_on_limit(monkeypatch, tmp_path):
    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"keep_long": "' + ("a" * 5001) + '"}', encoding="utf-8")

    out_dir = tmp_path / "tmp-output/elevenlabs"
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[openai]
api_key = "dummy-openai"

[hume_ai]
api_key = "dummy-hume"

[elevenlabs]
api_key = "dummy-eleven"
model = "eleven_multilingual_v2"
voice_name = "voice-id"
max_chars_limit = 5000

[voice]
input_file = "{progress.as_posix()}"
output_dir = "{out_dir.as_posix()}"
audio_format = "mp3"
allowed_regex = "^keep"
ignore_regex = ""
stop_after = 0
provider = "elevenlabs"
dry_run = true
        """,
        encoding="utf-8",
    )

    warnings: list[str] = []

    def _warn(message: str) -> None:
        warnings.append(message)

    monkeypatch.setattr(ai_voice.logger, "warning", _warn)

    ai_voice.generate_voice(
        config_path=config,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
        dry_run=True,
    )

    assert any("keep_long" in message for message in warnings)


def test_voice_dry_run_warns_on_non_octave2(monkeypatch, tmp_path):
    progress = tmp_path / "tmp-cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"keep_one": "Hello"}', encoding="utf-8")

    out_dir = tmp_path / "tmp-output/hume"
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[openai]
api_key = "dummy-openai"

[hume_ai]
api_key = "dummy-hume"
model = "octave"
voice_name = "ivan"
octave_version = "1"

[voice]
input_file = "{progress.as_posix()}"
output_dir = "{out_dir.as_posix()}"
audio_format = "mp3"
allowed_regex = "^keep"
ignore_regex = ""
stop_after = 0
provider = "hume_ai"
dry_run = true
        """,
        encoding="utf-8",
    )

    warnings: list[str] = []

    def _warn(message: str) -> None:
        warnings.append(message)

    monkeypatch.setattr(ai_voice.logger, "warning", _warn)

    ai_voice.generate_voice(
        config_path=config,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
        dry_run=True,
    )

    assert any("Default pricing assumes Octave 2" in message for message in warnings)


def test_voice_dry_run_summary_counts():
    worklist = [("a", "Hi"), ("b", "Hola")]
    summary = ai_voice._summarize_voice_dry_run(
        worklist,
        free_chars=3,
        rate_per_1k=0.1,
        currency="USD",
    )
    assert summary["total_chars"] == 6
    assert summary["free_chars"] == 3
    assert summary["billable_chars"] == 3
    assert summary["rate_per_1k"] == 0.1
    assert summary["estimated_cost"] == pytest.approx(0.0003)


def test_voice_stop_after_limits_worklist(monkeypatch, tmp_path, stub_runner):
    config_path = write_config(tmp_path)
    progress = write_progress(tmp_path)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=tmp_path / "out/hume",
        stop_after=1,
        allowed_regex="^keep",
    )

    worklist = stub_runner["worklist"]
    assert len(worklist) == 1


def test_generate_voice_missing_file(tmp_path):
    config_path = write_config(tmp_path)
    with pytest.raises(SystemExit):
        ai_voice.generate_voice(config_path=config_path, input_file=tmp_path / "none.json")


def test_voice_typer_command(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_progress(tmp_path)

    called = {}

    def fake_generate_voice(**kwargs):
        called["ok"] = kwargs

    monkeypatch.setattr(ai_voice, "generate_voice", fake_generate_voice)

    ai_voice.typer_command(
        cast(typer.Context, DummyCtx()),
        input_file=tmp_path / "tmp-cache/progress.json",
        output_dir=tmp_path / "out/hume",
        config_path=config_path,
        allowed_regex="^keep",
        ignore_regex="",
        stop_after=1,
        audio_format="mp3",
    )
    assert called["ok"]["input_file"] == tmp_path / "tmp-cache/progress.json"


def test_ai_voice_main_entry(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["voicebound-voice", "--help"])
    try:
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "src/commands/ai_voice.py"),
            run_name="__main__",
        )
    except SystemExit as exc:
        assert exc.code == 0
