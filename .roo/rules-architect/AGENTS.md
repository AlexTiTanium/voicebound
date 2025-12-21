# Architect Mode Rules (Non-Obvious Only)

- **Async Architecture**: The system is built around `anyio` and `TaskRunner`. New batch operations must fit this pattern (rate-limited, concurrent, retrying).
- **Provider Isolation**: Providers (OpenAI, Hume) should remain stateless. Configuration and rate limits are injected via `ProviderSettings`.
- **CLI Injection**: `typer` context (`ctx.obj`) is the dependency injection mechanism for global config. Do not bypass this by reading config directly in subcommands.
- **Progress Persistence**: State is persisted to a JSON file to allow resuming. This is a core architectural decision to handle long-running API tasks.
- **Error Handling Strategy**: `TaskRunner` captures exceptions and records them in `TaskOutcome`. It does not crash the process on individual task failures.
