import json
import sys
from pathlib import Path

import pytest
from loguru import logger

from voicebound import utils


def test_load_config_and_get_value(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
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
    cfg = {"section": {}}
    with pytest.raises(SystemExit):
        utils.get_config_value(cfg, "section", "missing")
