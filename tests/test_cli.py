import re
import runpy
from pathlib import Path

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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
