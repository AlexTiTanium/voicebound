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

## Configuration
- Copy the example config and add your API keys (do not commit secrets):
  - `cp config.example.toml config.toml`
  - Set `openai.api_key` for translation and `hume_ai.api_key` for voice in `config.toml`.

## ai_translate.py (translate strings)
- Run: `python ai_translate.py [path/to/strings.xml]`
  - Defaults to `strings.xml` in this folder.
  - Output: `out/values/strings.xml`
  - Progress cache: `.cache/progress.json`

## ai_voice.py (generate audio)
- Run: `python ai_voice.py`
  - Reads `.cache/progress.json`
  - Writes audio files to `out/hume/`

## License
- MIT (see `LICENSE`)
