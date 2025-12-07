"""Compatibility wrapper so legacy entry points importing voicebound.cli continue to work."""

from cli import app  # re-export current Typer app

__all__ = ["app"]

