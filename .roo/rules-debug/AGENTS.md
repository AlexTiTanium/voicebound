# Debug Mode Rules (Non-Obvious Only)

- **Config Resolution**: If config is missing, check `VOICEBOUND_CONFIG` env var first, then local `config.toml`, then XDG. `src/utils/__init__.py::resolve_config_path` logic is strict.
- **Logging Output**: Logs are forced to `stderr` by `src/utils/__init__.py::configure_logging`. If you don't see logs, check stderr redirection.
- **Progress vs Logs**: `stdout` is reserved for `rich` progress bars. Mixing print/logging to stdout will break the UI.
- **Test Imports**: `tests/conftest.py` modifies `sys.path` to include `src`. If tests fail with `ModuleNotFoundError`, ensure `conftest.py` is being picked up by pytest.
- **JSON Errors**: `load_json` silently attempts to repair trailing commas. If you suspect JSON corruption, check if the file was rewritten (it logs a warning).
- **Typer Errors**: `typer` swallows some exceptions. Use `--log-level DEBUG` to see full tracebacks if CLI commands fail silently.
