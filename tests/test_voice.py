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


def test_voice_stop_after_limits_worklist(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    # Extend progress to multiple entries
    progress = tmp_path / ".cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(
        '{"keep_one": "Hola mundo", "keep_two": "Hola mundo", "keep_three": "Hola mundo"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=tmp_path / "out/hume",
        stop_after=1,
        allowed_regex="^keep",
    )

    out_dir = tmp_path / "out/hume"
    files = list(out_dir.glob("*.mp3"))
    assert len(files) == 1


def test_voice_respects_existing_outputs(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    progress = write_progress(tmp_path)

    out_dir = tmp_path / "out/hume"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pretend keep_one already exists
    (out_dir / "keep_one.mp3").write_bytes(b"existing")

    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
    )

    # keep_one should remain untouched, keep_two generated
    assert (out_dir / "keep_one.mp3").read_bytes() == b"existing"
    assert (out_dir / "skip_me.mp3").exists() is False
    # skip_me and other filtered; only keep_one existed
    assert (out_dir / "keep_two.mp3").exists() is False


def test_attempt_send_stops_on_event(monkeypatch, tmp_path):
    called = {"count": 0}

    def _send(headers, payload):
        called["count"] += 1
        return DummyResponse()

    monkeypatch.setattr(ai_voice, "send_request", _send)
    stop_event = ai_voice.Event()
    stop_event.set()

    success, msg = ai_voice.attempt_send(
        payload={"x": "y"},
        headers={},
        out_path=tmp_path / "x.mp3",
        rate_limiter=ai_voice.RateLimiter(0),
        max_retries=3,
        backoff_seconds=(0,),
        stop_event=stop_event,
    )

    assert success is False
    assert msg == "interrupted"
    assert called["count"] == 0
