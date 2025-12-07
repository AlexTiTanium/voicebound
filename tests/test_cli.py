import re
import runpy
from pathlib import Path

import pytest
import typer

import cli
from typer.testing import CliRunner

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
        runpy.run_path(cli_path, run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0


def test_version_callback_prints_and_exits(capsys):
    with pytest.raises(typer.Exit):
        cli.version_callback(True)
    assert cli.APP_VERSION in capsys.readouterr().out


def test_main_without_subcommand_uses_help(capsys):
    ctx = DummyCtx()
    with pytest.raises(typer.Exit):
        cli.main(ctx, config_path=None, log_level="INFO", no_color=True, version=False)
    out = capsys.readouterr().out
    assert "HELP" in out
    assert ctx.obj["color"] is False
