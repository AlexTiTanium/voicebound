import runpy
from pathlib import Path

from typer.testing import CliRunner

from voicebound.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    assert "translate" in result.stdout
    assert "voice" in result.stdout


def test_cli_translate_help():
    result = runner.invoke(app, ["translate", "--help"], color=False)
    assert result.exit_code == 0
    assert "--input-file" in result.stdout


def test_cli_voice_help():
    result = runner.invoke(app, ["voice", "--help"], color=False)
    assert result.exit_code == 0
    assert "--output-dir" in result.stdout


def test_cli_main_entry(monkeypatch):
    cli_path = Path(__file__).resolve().parents[1] / "src/voicebound/cli.py"
    monkeypatch.setattr("sys.argv", ["voicebound", "--help"])
    try:
        runpy.run_path(cli_path, run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
