import importlib.metadata
from pathlib import Path

import typer

from commands import ai_translate, ai_voice

try:
    APP_VERSION = importlib.metadata.version("voicebound")
except importlib.metadata.PackageNotFoundError:
    APP_VERSION = "0.0.0"

app = typer.Typer(
    help="Voicebound CLI utilities for translating text and generating audio.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """
    Print the application version and exit when the flag is set.

    Args:
        value: User-supplied flag from the CLI option callback.
    """
    if value:
        typer.echo(APP_VERSION)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        "-c",
        help="Path to config.toml (env VOICEBOUND_CONFIG or defaults if omitted).",
    ),
    log_level: str | None = typer.Option(
        None, "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR)."
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored log output."),
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """
    Global options applied to all subcommands.

    Args:
        ctx: Typer context populated by the CLI runtime.
        config_path: User-supplied config path override (CLI flag).
        log_level: User-supplied log verbosity (CLI flag).
        no_color: User-supplied flag to disable colored logging.
        version: User-supplied flag to print version and exit.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["log_level"] = log_level
    ctx.obj["color"] = not no_color
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


app.command("translate", help="Translate strings.xml into a target language.")(
    ai_translate.typer_command
)
app.command("voice", help="Generate audio from translated text.")(ai_voice.typer_command)


if __name__ == "__main__":
    app()
