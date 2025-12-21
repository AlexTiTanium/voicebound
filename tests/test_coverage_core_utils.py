import re
from pathlib import Path

import anyio
import pytest

import utils
from core.command_context import build_tasks, make_command_context, run_with_progress
from core.summary_reporter import SummaryReporter
from core.task_runner import RetryConfig, RunnerConfig, TaskHooks, TaskRunner, TaskSpec
from utils.command_utils import (
    OutcomeCollector,
    ProgressReporter,
    build_voice_worklist,
    derive_concurrency,
    load_strings,
    log_retry,
)


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[openai]
api_key = "dummy"

[hume_ai]
api_key = "dummy"
        """,
        encoding="utf-8",
    )
    return path


def test_make_command_context(tmp_path):
    cfg = _write_config(tmp_path)
    ctx = make_command_context(
        config_path=cfg,
        provider_key="openai",
        default_model="gpt-5-nano",
        default_rpm=60,
        concurrency_override=2,
    )
    assert ctx.provider.api_key == "dummy"
    assert ctx.provider.concurrency == 2


def test_build_tasks_wraps_coroutines():
    async def coro_one():
        return "one"

    specs = build_tasks([("one", coro_one)])
    result = anyio.run(specs[0].coro_factory)
    assert result == "one"


def test_run_with_progress_success_default():
    class DummyRunner:
        def __init__(self):
            self.hooks = None
            self.config = RunnerConfig(
                name="test",
                rpm=1,
                concurrency=1,
                retry=RetryConfig(attempts=1, backoff_base=0.0, backoff_max=0.0, jitter=False),
            )

        async def run(self, specs):
            for spec in specs:
                await self.hooks.on_success(spec, "ok")

    runner = DummyRunner()
    summary = SummaryReporter("test")
    specs = [TaskSpec(task_id="ok", coro_factory=lambda: None)]
    anyio.run(run_with_progress, "test", 1, runner, specs, summary)
    assert summary.successes == 1


def test_run_with_progress_failure_default():
    class DummyRunner:
        def __init__(self):
            self.hooks = None
            self.config = RunnerConfig(
                name="test",
                rpm=1,
                concurrency=1,
                retry=RetryConfig(attempts=1, backoff_base=0.0, backoff_max=0.0, jitter=False),
            )

        async def run(self, specs):
            for spec in specs:
                await self.hooks.on_failure(spec, ValueError("fail"))

    runner = DummyRunner()
    summary = SummaryReporter("test")
    specs = [TaskSpec(task_id="fail", coro_factory=lambda: None)]
    anyio.run(run_with_progress, "test", 1, runner, specs, summary)
    assert "fail" in summary.failures


def test_run_with_progress_retry_default_logs(monkeypatch):
    class DummyRunner:
        def __init__(self):
            self.hooks = None
            self.config = RunnerConfig(
                name="test",
                rpm=1,
                concurrency=1,
                retry=RetryConfig(attempts=2, backoff_base=0.0, backoff_max=0.0, jitter=False),
            )

        async def run(self, specs):
            for spec in specs:
                self.hooks.on_retry(spec, 1, 0.0)
                await self.hooks.on_success(spec, "ok")

    runner = DummyRunner()
    calls: list[str] = []

    def _fake_warning(message: str) -> None:
        calls.append(message)

    monkeypatch.setattr("core.command_context.logger.warning", _fake_warning)
    summary = SummaryReporter("test")
    specs = [TaskSpec(task_id="flaky", coro_factory=lambda: None)]
    anyio.run(run_with_progress, "test", 1, runner, specs, summary)
    assert calls


def test_run_with_progress_retry_callback():
    class DummyRunner:
        def __init__(self):
            self.hooks = None
            self.config = RunnerConfig(
                name="test",
                rpm=1,
                concurrency=1,
                retry=RetryConfig(attempts=2, backoff_base=0.0, backoff_max=0.0, jitter=False),
            )

        async def run(self, specs):
            for spec in specs:
                self.hooks.on_retry(spec, 1, 0.0)
                await self.hooks.on_success(spec, "ok")

    runner = DummyRunner()
    summary = SummaryReporter("test")
    specs = [TaskSpec(task_id="retry", coro_factory=lambda: None)]
    calls: list[int] = []

    def _retry_cb(_spec, attempt, _sleep_for):
        calls.append(attempt)

    async def _run():
        return await run_with_progress(
            "test",
            1,
            runner,
            specs,
            summary,
            retry_cb=_retry_cb,
        )

    anyio.run(_run)
    assert calls == [1]


def test_task_runner_retry_error_path(monkeypatch):
    from tenacity import RetryError

    cfg = RunnerConfig(
        name="runner",
        rpm=1,
        concurrency=1,
        retry=RetryConfig(attempts=1, backoff_base=0.0, backoff_max=0.0, jitter=False),
    )
    calls: list[str] = []

    async def on_failure(_spec, _exc):
        calls.append("failed")

    runner = TaskRunner(cfg, TaskHooks(on_failure=on_failure))

    class DummyAttempt:
        def exception(self):
            return ValueError("boom")

    class RaisingRetrying:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RetryError(DummyAttempt())

    monkeypatch.setattr(runner, "_make_retrying", lambda _spec: RaisingRetrying())

    async def boom():
        raise ValueError("boom")

    outcomes: list = []

    async def _run():
        await runner._run_single(TaskSpec(task_id="boom", coro_factory=boom), outcomes, anyio.Lock())

    anyio.run(_run)
    assert outcomes[0].ok is False
    assert calls == ["failed"]


def test_task_runner_retry_hook_and_async_success():
    cfg = RunnerConfig(
        name="runner",
        rpm=1,
        concurrency=1,
        retry=RetryConfig(attempts=2, backoff_base=0.0, backoff_max=0.0, jitter=False),
    )
    retry_calls: list[int] = []

    def on_retry(_spec, attempt, _sleep_for):
        retry_calls.append(attempt)

    async def on_success(_spec, _result):
        return None

    runner = TaskRunner(cfg, TaskHooks(on_retry=on_retry, on_success=on_success))
    handler = runner._before_sleep(TaskSpec(task_id="flaky", coro_factory=lambda: None))

    class DummyNextAction:
        sleep = 0.0

    class DummyState:
        attempt_number = 1
        next_action = DummyNextAction()

    handler(DummyState())
    assert retry_calls == [1]


def test_task_runner_before_sleep_no_retry():
    cfg = RunnerConfig(
        name="runner",
        rpm=1,
        concurrency=1,
        retry=RetryConfig(attempts=1, backoff_base=0.0, backoff_max=0.0, jitter=False),
    )
    runner = TaskRunner(cfg, TaskHooks())
    handler = runner._before_sleep(TaskSpec(task_id="none", coro_factory=lambda: None))

    class DummyState:
        attempt_number = 1
        next_action = None

    handler(DummyState())


def test_task_runner_maybe_call_none():
    anyio.run(TaskRunner._maybe_call, None)


def test_task_runner_maybe_call_coroutine():
    async def _callback(value):
        return value

    anyio.run(TaskRunner._maybe_call, _callback, "ok")


def test_derive_concurrency_paths(monkeypatch):
    monkeypatch.setattr("utils.command_utils.os.cpu_count", lambda: 2)
    assert derive_concurrency(10) == 1
    assert derive_concurrency(10, override=4) == 4


def test_load_strings_root_missing(monkeypatch, tmp_path):
    class DummyTree:
        def getroot(self):
            return None

    monkeypatch.setattr("utils.command_utils.ET.parse", lambda _path: DummyTree())
    with pytest.raises(ValueError):
        load_strings(tmp_path / "fake.xml")


def test_build_voice_worklist_branches(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "file_exists.mp3").write_text("data", encoding="utf-8")
    progress = {
        "none": None,
        "existing": "text",
        "ignored": "text",
        "not_allowed": "text",
        "file_exists": "text",
        "ok": "text",
    }
    allowed = re.compile(r"^(file_exists|ok)$")
    ignore = re.compile(r"^ignored$")
    worklist = build_voice_worklist(
        progress,
        allowed,
        ignore,
        existing_outputs={"existing"},
        stop_after=0,
        output_dir=output_dir,
        audio_format="mp3",
    )
    assert worklist == [("ok", "text")]


def test_build_voice_worklist_stop_after(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    progress = {"a": "text", "b": "text"}
    allowed = re.compile(r"^")
    worklist = build_voice_worklist(
        progress,
        allowed,
        None,
        existing_outputs=set(),
        stop_after=1,
        output_dir=output_dir,
        audio_format="mp3",
    )
    assert len(worklist) == 1


def test_progress_reporter_context_and_advance():
    reporter = ProgressReporter("Testing", total=1)
    reporter.advance()
    with reporter as progress:
        progress.advance()


def test_outcome_collector_and_log_retry():
    collector = OutcomeCollector("name")
    collector.record_success("ok")
    collector.record_failure("bad")
    assert collector.successes == ["ok"]
    assert collector.failures == ["bad"]
    log_retry("translate", "task", 1, 3, None)


def test_summary_reporter_paths():
    reporter = SummaryReporter("translate")
    reporter.record_translation("translated", "a")
    reporter.record_translation("skipped", "b")
    reporter.record_translation("loaded", "c")
    reporter.record_translation("ignored", "d")
    reporter.record_translation("empty", "e")
    reporter.record_translation("error", "bad")
    reporter.record_success("ok")
    reporter.record_failure("fail")
    reporter.log_translation(output_path="out.xml")
    reporter.log_voice(["ok1"], ["bad1"], skipped=2)


def test_load_json_repairs_trailing_comma(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a": 1,}', encoding="utf-8")
    data = utils.load_json(path)
    assert data == {"a": 1}
    assert path.read_text(encoding="utf-8").strip().endswith("}")


def test_load_json_repair_failure_returns_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a": 1,]', encoding="utf-8")
    data = utils.load_json(path, default={"fallback": True})
    assert data == {"fallback": True}


def test_load_json_unrepaired_returns_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    data = utils.load_json(path, default={"fallback": True})
    assert data == {"fallback": True}


def test_write_json_cleans_up_tmp_file(monkeypatch, tmp_path):
    path = tmp_path / "data.json"
    tmp_file = tmp_path / "data.json.tmp"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(utils.json, "dump", _boom)
    with pytest.raises(RuntimeError):
        utils.write_json(path, {"a": 1})
    assert not tmp_file.exists()
