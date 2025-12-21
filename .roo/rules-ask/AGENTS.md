# Ask Mode Rules (Non-Obvious Only)

- **Core vs Utils**: `src/core` contains business logic infrastructure (TaskRunner, CommandContext), while `src/utils` contains generic helpers (JSON, XML, Logging).
- **XML Structure**: The project specifically targets Android `strings.xml` format. `load_strings` returns an `ElementTree` and `Element` root.
- **Provider Abstraction**: `src/apis/` contains provider implementations (OpenAI, Hume), but they are configured via `src/utils/command_utils.py::ProviderSettings`.
- **Progress File**: Progress is tracked in a JSON file (default `.cache/progress.json`), not a database.
- **Config Format**: Config is TOML, not JSON or YAML.
