# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project-Specific Commands
- **Run**: `uv run voicebound` (entry point is `src/cli.py`)
- **Test**: `uv run pytest` (requires `src` in `sys.path`, handled by `tests/conftest.py`)
- **Lint/Format**: `uv run ruff check .` / `uv run ruff format .`
- **Type Check**: `uv run ty check`

## Critical Patterns (Non-Obvious)
- **Task Runner**: All batch operations MUST use `src/core/task_runner.py::TaskRunner` for rate limiting/retries.
- **Logging**: `src/utils/__init__.py::configure_logging` forces logs to `stderr` to keep `stdout` clean for `rich` progress bars.
- **Config**: Resolution order is explicit > `VOICEBOUND_CONFIG` > local `config.toml` > XDG.
- **JSON**: `src/utils/__init__.py::load_json` auto-repairs trailing commas (non-standard).
- **CLI State**: `typer` commands receive global config via `ctx.obj` populated in `src/cli.py::main`.

## Gotchas
- **Tests**: `tests/conftest.py` manually inserts `src` into `sys.path`; tests may fail if run without this fixture.
- **Progress**: Use `src/utils/command_utils.py::ProgressReporter` context manager, NOT raw `rich.Progress`.
