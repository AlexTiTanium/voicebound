from pathlib import Path

import pytest

from voicebound.commands import ai_voice


class DummyResponse:
    def __init__(self, status_code=200, content=b"audio"):
        self.status_code = status_code
        self.content = content
        self.text = "ok"


def write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[hume_ai]
api_key = "dummy-hume"
model = "octave"
voice_name = "ivan"
octave_version = "2"
split_utterances = true

[voice]
input_file = ".cache/progress.json"
output_dir = "out/hume"
audio_format = "mp3"
allowed_regex = "^keep"
ignore_regex = "skip"
stop_after = 0
max_workers = 2
request_delay_seconds = 0.0
max_retries = 1
backoff_seconds = [0]
target_language = "Spanish"
provider = "HUME_AI"
        """,
        encoding="utf-8",
    )
    return config


def write_progress(tmp_path: Path) -> Path:
    progress = tmp_path / ".cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(
        '{"keep_one": "Hola mundo", "skip_me": "ignored", "other": "nope"}',
        encoding="utf-8",
    )
    return progress


def test_voice_generates_only_allowed(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_progress(tmp_path)

    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=tmp_path / ".cache/progress.json",
        output_dir=tmp_path / "out/hume",
    )

    out_dir = tmp_path / "out/hume"
    assert (out_dir / "keep_one.mp3").exists()
    assert not (out_dir / "skip_me.mp3").exists()
    assert not (out_dir / "other.mp3").exists()


def test_voice_interrupts_in_futures(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_progress(tmp_path)

    def _raise(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(ai_voice, "send_request", _raise)
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    with pytest.raises(SystemExit) as excinfo:
        ai_voice.generate_voice(
            config_path=config_path,
            input_file=tmp_path / ".cache/progress.json",
            output_dir=tmp_path / "out/hume",
        )

    assert excinfo.value.code == 130
