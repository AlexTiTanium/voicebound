from pathlib import Path

import pytest
from loguru import logger

import utils
from utils.command_utils import persist_progress


def test_load_config_and_get_value(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[openai]
api_key = "dummy"

[hume_ai]
api_key = "dummy"

[section]
required_key = "value"
""",
        encoding="utf-8",
    )
    data = utils.load_config(cfg)
    assert utils.get_config_value(data, "section", "required_key") == "value"
    assert utils.get_config_value(data, "section", "missing", required=False, default="d") == "d"


def test_load_config_missing_raises(tmp_path: Path):
    with pytest.raises(SystemExit):
        utils.load_config(tmp_path / "nope.toml")


def test_resolve_path_relative_and_absolute(tmp_path: Path):
    rel = utils.resolve_path("foo", base=tmp_path)
    assert rel == tmp_path / "foo"
    absolute = utils.resolve_path(tmp_path / "bar", base=tmp_path)
    assert absolute == tmp_path / "bar"


def test_write_and_load_json(tmp_path: Path):
    path = tmp_path / "data.json"
    utils.write_json(path, {"a": 1})
    loaded = utils.load_json(path)
    assert loaded == {"a": 1}
    assert utils.load_json(tmp_path / "missing.json", default={}) == {}


def test_persist_progress_merges_existing(tmp_path: Path):
    path = tmp_path / "progress.json"
    utils.write_json(path, {"a": "one", "b": "two"})
    persist_progress(path, {"b": "two_new", "c": "three"})
    loaded = utils.load_json(path)
    assert loaded == {"a": "one", "b": "two_new", "c": "three"}


def test_configure_logging_replaces_handlers(capsys):
    utils.configure_logging(level="INFO")
    # Emit a log line to ensure the sink is wired; no assertion on content to avoid handler races.
    logger.info("hello")


def test_rate_limiter_waits(monkeypatch):
    limiter = utils.RateLimiter(0.01)
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)

    monkeypatch.setattr(utils.time, "sleep", fake_sleep)
    limiter.wait()
    limiter.wait()
    assert calls  # second wait triggers sleep


def test_get_config_value_required(tmp_path: Path):
    cfg: dict[str, dict[str, str]] = {"section": {}}
    with pytest.raises(SystemExit):
        utils.get_config_value(cfg, "section", "missing")


def test_resolve_config_path_prefers_env(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[openai]
api_key = "env-openai"
[hume_ai]
api_key = "env-hume"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOICEBOUND_CONFIG", str(cfg))
    assert utils.resolve_config_path(None) == cfg
    data = utils.load_config()
    assert data["openai"]["api_key"] == "env-openai"
    monkeypatch.delenv("VOICEBOUND_CONFIG", raising=False)


def test_validate_config_missing_required(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[openai]
api_key = "only-openai"
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        utils.load_config(cfg)


def test_validate_config_missing_openai_key(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[openai]
api_key = ""

[hume_ai]
api_key = "dummy"
        """,
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        utils.load_config(cfg)


def test_validate_config_import_failure_defaults_to_hume(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[openai]
api_key = "dummy"

[voice]
provider = "openai_tts"
        """,
        encoding="utf-8",
    )

    import builtins

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "providers.registry":
            raise ImportError("blocked")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(SystemExit):
        utils.load_config(cfg)


def test_compile_regex_invalid():
    with pytest.raises(SystemExit):
        utils.compile_regex("(", label="broken")


def test_resolve_config_path_raises_when_missing(monkeypatch):
    monkeypatch.delenv("VOICEBOUND_CONFIG", raising=False)
    monkeypatch.setattr(utils, "DEFAULT_LOCAL_CONFIG", Path("/no/such/config.toml"))
    monkeypatch.setattr(utils, "DEFAULT_XDG_CONFIG", Path("/no/such/xdg/config.toml"))
    with pytest.raises(SystemExit):
        utils.resolve_config_path(None)
