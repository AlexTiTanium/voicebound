from __future__ import annotations

import typer

from commands import ai_translate, ai_voice

app = typer.Typer(help="Voicebound CLI utilities.")

app.command("translate")(ai_translate.typer_command)
app.command("voice")(ai_voice.typer_command)


if __name__ == "__main__":
    app()
