# Voicebound

Utilities for translating and generating audio from text assets.

### Install Python 3
- **macOS**: `brew install python` (Homebrew). Verify with `python3 --version`.
- **Windows**: Download installer from https://www.python.org/downloads/windows/ (check “Add python.exe to PATH” during setup). Verify with `python --version`.

## Setup
- Ensure Python 3 is installed.
- From this folder:
  - `./install.sh` — creates/uses `venv`, installs dependencies, and opens a shell with the venv activated.
  - `./install.sh --no-activate` — installs dependencies only.
  - Or install the package in editable mode for the `voicebound` CLI: `pip install -e .`

## Configuration
- Copy the example config and add your API keys (do not commit secrets):
  - `cp config.example.toml config.toml`
  - Set `openai.api_key` for translation and `hume_ai.api_key` for voice in `config.toml`.

## CLI usage
- Run `voicebound --help` for the full command list (after `pip install -e .`).
- Translate strings: `voicebound translate --input-file strings.xml`
  - Default output: `out/values/strings.xml`
  - Progress cache: `.cache/progress.json`
- Generate audio: `voicebound voice`
  - Reads translations from `.cache/progress.json`
  - Writes audio files to `out/hume/`

## Project layout
- Command implementations live in `src/voicebound/commands/`.
- Shared helpers and logging live in `src/voicebound/`.

## License
- MIT (see `LICENSE`)
