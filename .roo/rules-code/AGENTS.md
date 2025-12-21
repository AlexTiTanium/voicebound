# Code Mode Rules (Non-Obvious Only)

- **Task Execution**: NEVER implement raw loops for API calls. Use `src/core/task_runner.py::TaskRunner` which handles `aiolimiter` and `tenacity` retries.
- **Command Context**: Initialize commands using `src/core/command_context.py::make_command_context` to ensure consistent config/logging setup.
- **Progress Bars**: MUST use `src/utils/command_utils.py::ProgressReporter` as a context manager. Direct `rich.Progress` usage will conflict with logging.
- **JSON Handling**: Use `src/utils/__init__.py::load_json` for reading JSON; it includes a custom regex-based repair for trailing commas.
- **XML Parsing**: Use `src/utils/command_utils.py::load_strings` for `strings.xml` files; it returns `(tree, root)`.
- **Async Concurrency**: `TaskRunner` uses `anyio.Semaphore` and `anyio.create_task_group`. Do not use `asyncio.gather` directly for batched tasks.
- **Typer Context**: Access global flags (config path, log level) via `ctx.obj["config_path"]` etc. in `typer` commands.
