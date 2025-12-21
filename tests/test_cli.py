import re
import runpy
from pathlib import Path
from typing import cast

import pytest
import typer
from typer.testing import CliRunner

import cli
from cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class DummyCtx:
    def __init__(self):
        self.obj = {}
        self.invoked_subcommand = None

    def ensure_object(self, typ):
        self.obj = typ()
        return self.obj

    def get_help(self):
        return "HELP"


def test_cli_help():
    result = runner.invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "translate" in output
    assert "voice" in output


def test_cli_translate_help():
    result = runner.invoke(app, ["translate", "--help"], color=False)
    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "--input-file" in output


def test_cli_voice_help():
    result = runner.invoke(app, ["voice", "--help"], color=False)
    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "--output-dir" in output


def test_cli_main_entry(monkeypatch):
    cli_path = Path(__file__).resolve().parents[1] / "src/cli.py"
    monkeypatch.setattr("sys.argv", ["voicebound", "--help"])
    try:
        runpy.run_path(str(cli_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0


def test_version_callback_prints_and_exits(capsys):
    with pytest.raises(typer.Exit):
        cli.version_callback(True)
    assert cli.APP_VERSION in capsys.readouterr().out


def test_main_without_subcommand_uses_help(capsys):
    ctx = DummyCtx()
    with pytest.raises(typer.Exit):
        cli.main(
            cast(typer.Context, ctx),
            config_path=None,
            log_level="INFO",
            no_color=True,
            version=False,
        )
    out = capsys.readouterr().out
    assert "HELP" in out
    assert ctx.obj["color"] is False


def test_cli_translate_smoke(monkeypatch, tmp_path):
    called = {}

    def fake_translate_strings(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("cli.ai_translate.translate_strings", fake_translate_strings)

    cfg = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        [
            "--config-path",
            str(cfg),
            "--log-level",
            "DEBUG",
            "--no-color",
            "translate",
            "--input-file",
            str(tmp_path / "in.xml"),
            "--output-file",
            str(tmp_path / "out.xml"),
            "--allowed-regex",
            "^keep",
            "--ignore-regex",
            "skip",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert called["input_file"] == Path(tmp_path / "in.xml")
    assert called["output_file"] == Path(tmp_path / "out.xml")
    assert called["allowed_regex"] == "^keep"
    assert called["ignore_regex"] == "skip"
    assert called["dry_run"] is True
    assert called["config_path"] == cfg
    assert called["log_level"] == "DEBUG"
    assert called["color"] is False


def test_cli_voice_smoke(monkeypatch, tmp_path):
    called = {}

    def fake_generate_voice(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("cli.ai_voice.generate_voice", fake_generate_voice)

    cfg = tmp_path / "config.toml"
    progress = tmp_path / "tmp-cache/progress.json"
    out_dir = tmp_path / "tmp-output"
    result = runner.invoke(
        app,
        [
            "--config-path",
            str(cfg),
            "--no-color",
            "voice",
            "--input-file",
            str(progress),
            "--output-dir",
            str(out_dir),
            "--provider",
            "HUME_AI",
            "--allowed-regex",
            "^keep",
            "--ignore-regex",
            "skip",
            "--stop-after",
            "2",
            "--audio-format",
            "wav",
        ],
    )
    assert result.exit_code == 0
    assert called["input_file"] == Path(progress)
    assert called["output_dir"] == Path(out_dir)
    assert called["provider"] == "HUME_AI"
    assert called["allowed_regex"] == "^keep"
    assert called["ignore_regex"] == "skip"
    assert called["stop_after"] == 2
    assert called["audio_format"] == "wav"
    assert called["config_path"] == cfg
    assert called["color"] is False
