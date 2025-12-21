# Voicebound

Utilities for translating Android-style `strings.xml` files and generating TTS audio assets.

### Install Python 3
- **macOS**: `brew install python` (Homebrew). Verify with `python3 --version`.
- **Windows**: Download installer from https://www.python.org/downloads/windows/ (check “Add python.exe to PATH” during setup). Verify with `python --version`.

## Quickstart
- Ensure Python 3.10+ is installed.
- Install uv: https://docs.astral.sh/uv/
- Create the environment and install deps: `uv sync --dev`
- Run translate: `uv run voicebound translate --input-file strings.xml --config-path config.toml`
- Run voice: `uv run voicebound voice --config-path config.toml`

## Configuration
- Start from the sample: `cp config.example.toml config.toml` and add your API keys (do not commit secrets).
- Required keys: `[openai].api_key` for translation, `[hume_ai].api_key` for voice.

## CLI usage
- Global flags: `--config-path/-c`, `--log-level/-l`, `--no-color`, `--version`.
- Translate (OpenAI):
  - `voicebound translate --input-file strings.xml --output-file out/values/strings.xml`
  - Common overrides: `--allowed-regex "^keep"`, `--ignore-regex "skip"`, `--dry-run` (estimates tokens only), `--target-language "Spanish"`.
- Voice (Hume):
  - `voicebound voice --input-file .cache/progress.json --output-dir out/hume`
  - Common overrides: `--allowed-regex "^keep"`, `--ignore-regex "skip"`, `--stop-after 10`, `--audio-format wav`, `--provider HUME_AI`.
- Completion/help: `voicebound --help`, `voicebound translate --help`, `voicebound voice --help`.

## Project layout
- CLI entrypoint: `src/cli.py` (console script `voicebound`).
- Commands: `src/commands/ai_translate.py`, `src/commands/ai_voice.py`.
- Shared helpers: `src/utils.py`.

## Tests
- Run unit/integration suite: `uv run pytest`
- Coverage locally: `uv run pytest --cov=src`

## Lint/format
- Ruff lint: `uv run ruff check .`
- Ruff format: `uv run ruff format .`
- Ty: `uv run ty check`

## Troubleshooting
- “Config not found”: verify `--config-path` or `VOICEBOUND_CONFIG` and that `config.toml` exists.
- “Missing required config keys”: set `[openai].api_key` and `[hume_ai].api_key` (non-empty).
- Rate limits / retries: tune `[voice].request_delay_seconds`, `[voice].backoff_seconds`, `[voice].jitter_fraction`, `[voice].max_elapsed_seconds`.
- Interrupts: Ctrl+C stops translation/voice quickly; partial progress remains in `.cache/progress.json`.
- Logs vs progress: logs go to stderr; progress bars use stdout. Add `--no-color` if your terminal strips ANSI.

## License
- MIT (see `LICENSE`)
