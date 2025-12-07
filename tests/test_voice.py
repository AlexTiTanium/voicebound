import runpy
from pathlib import Path

import requests

import pytest

from voicebound.commands import ai_voice


class DummyResponse:
    def __init__(self, status_code=200, content=b"audio", text="ok"):
        self.status_code = status_code
        self.content = content
        self.text = text


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


def test_voice_send_request(monkeypatch):
    called = {}

    def fake_post(url, headers, json, timeout):
        called["ok"] = True
        return DummyResponse()

    monkeypatch.setattr(ai_voice.requests, "post", fake_post)
    resp = ai_voice.send_request({}, {})
    assert resp.status_code == 200
    assert called["ok"]


def test_handle_entry_error_branch(monkeypatch, tmp_path):
    out_path = tmp_path / "x.mp3"
    called = {}

    def fake_attempt_send(**kwargs):
        called["ok"] = True
        return False, "err"

    monkeypatch.setattr(ai_voice, "attempt_send", fake_attempt_send)
    ai_voice.handle_entry(
        key="k",
        text="t",
        headers={},
        rate_limiter=ai_voice.RateLimiter(0),
        out_path=out_path,
        max_retries=1,
        backoff_seconds=(0,),
        model="m",
        voice_name="v",
        provider="p",
        audio_format="mp3",
        split_utterances=False,
        target_language="es",
        octave_version="2",
        stop_event=ai_voice.Event(),
    )
    assert called["ok"]


def test_attempt_send_non_200(monkeypatch, tmp_path):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ai_voice.time, "sleep", fake_sleep)
    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse(500, b"", "bad"))
    success, msg = ai_voice.attempt_send(
        payload={},
        headers={},
        out_path=tmp_path / "x.mp3",
        rate_limiter=ai_voice.RateLimiter(0),
        max_retries=2,
        backoff_seconds=(0, 0),
        stop_event=ai_voice.Event(),
    )
    assert success is False
    assert msg.startswith("HTTP 500")
    assert sleeps  # backoff happened


def test_attempt_send_request_exception(monkeypatch, tmp_path):
    def _raise(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(ai_voice, "send_request", _raise)
    success, msg = ai_voice.attempt_send(
        payload={},
        headers={},
        out_path=tmp_path / "x.mp3",
        rate_limiter=ai_voice.RateLimiter(0),
        max_retries=1,
        backoff_seconds=(0,),
        stop_event=ai_voice.Event(),
    )
    assert success is False
    assert "boom" in msg


def test_generate_voice_missing_file(tmp_path):
    config_path = write_config(tmp_path)
    with pytest.raises(SystemExit):
        ai_voice.generate_voice(config_path=config_path, input_file=tmp_path / "none.json")


def test_generate_voice_existing_out_path(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    progress = write_progress(tmp_path)
    out_dir = tmp_path / "out/hume"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keep_one.mp3").write_bytes(b"existing")
    (out_dir / "keep_two.mp3").write_bytes(b"existing2")

    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
        audio_format="wav",
    )

    assert (out_dir / "keep_one.mp3").read_bytes() == b"existing"
    assert (out_dir / "keep_two.mp3").read_bytes() == b"existing2"
    # wav files were treated as existing and skipped
    (out_dir / "keep_one.wav").write_bytes(b"existing3")
    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
        audio_format="wav",
    )
    assert (out_dir / "keep_one.wav").read_bytes() == b"existing3"


def test_generate_voice_out_path_exists_without_existing_outputs(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    progress = tmp_path / ".cache/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"keep_one":"hola"}', encoding="utf-8")
    out_dir = tmp_path / "out/hume"
    out_dir.mkdir(parents=True, exist_ok=True)
    # only wav exists; mp3 set empty so branch for out_path.exists hit
    (out_dir / "keep_one.wav").write_bytes(b"existing")

    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    ai_voice.generate_voice(
        config_path=config_path,
        input_file=progress,
        output_dir=out_dir,
        allowed_regex="^keep",
        audio_format="wav",
    )

    assert (out_dir / "keep_one.wav").read_bytes() == b"existing"


def test_generate_voice_outer_keyboard_interrupt(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_progress(tmp_path)

    class BoomExecutor:
        def __init__(self, *args, **kwargs):
            self.tasks = []

        def submit(self, *args, **kwargs):
            raise KeyboardInterrupt()

        def shutdown(self, *args, **kwargs):
            return None

    monkeypatch.setattr(ai_voice, "ThreadPoolExecutor", BoomExecutor)
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(SystemExit):
        ai_voice.generate_voice(
            config_path=config_path,
            input_file=tmp_path / ".cache/progress.json",
            output_dir=tmp_path / "out/hume",
        )


def test_voice_typer_command(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    write_progress(tmp_path)

    monkeypatch.setattr(ai_voice, "send_request", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ai_voice.RateLimiter, "wait", lambda self: None)

    ai_voice.typer_command(
        input_file=tmp_path / ".cache/progress.json",
        output_dir=tmp_path / "out/hume",
        config_path=config_path,
        allowed_regex="^keep",
        ignore_regex="",
        stop_after=1,
        audio_format="mp3",
    )


def test_ai_voice_main_entry(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["voicebound-voice", "--help"])
    try:
        runpy.run_path(
            Path(__file__).resolve().parents[1] / "src/voicebound/commands/ai_voice.py",
            run_name="__main__",
        )
    except SystemExit as exc:
        assert exc.code == 0
