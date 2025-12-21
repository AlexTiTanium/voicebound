import re
from pathlib import Path
from threading import Lock
from xml.etree.ElementTree import Element

import anyio
import pytest

from apis.translation_api import (
    TranslationFilters,
    TranslationProgress,
    TranslationService,
    TranslationSettings,
)
from apis.voice_api import VoiceService, VoiceSettings
from core.summary_reporter import SummaryReporter
from core.task_runner import RetryConfig
from utils.command_utils import ProviderSettings


class DummyTranslationProvider:
    def translate_text(self, text: str, model: str, target_language: str) -> str:
        return f"{text}-{target_language}"


def _make_node(name: str, text: str | None) -> Element:
    node = Element("string", {"name": name})
    node.text = text
    return node


def _make_provider_settings() -> ProviderSettings:
    return ProviderSettings(
        api_key="dummy",
        model="gpt-5-nano",
        rpm=1000,
        concurrency=1,
        retry=RetryConfig(attempts=1, backoff_base=0.0, backoff_max=0.0, jitter=False),
    )


def test_process_node_branches(tmp_path):
    provider = DummyTranslationProvider()
    service = TranslationService(provider)
    filters = TranslationFilters(
        allowed_pattern=re.compile(r"^keep"),
        ignore_pattern=re.compile(r"^skip"),
    )
    progress = TranslationProgress(
        done={"other": "cached"},
        progress_file=tmp_path / "progress.json",
        progress_lock=Lock(),
    )
    settings = TranslationSettings(
        model="gpt-5-nano",
        target_language="Spanish",
        dry_run=False,
        count_tokens_enabled=False,
    )

    ignored = service._process_node(
        _make_node("skip_one", "Hello"),
        filters=filters,
        progress=progress,
        settings=settings,
        encoding=None,
    )
    assert ignored[2] == "ignored"

    empty = service._process_node(
        _make_node("keep_empty", ""),
        filters=filters,
        progress=progress,
        settings=settings,
        encoding=None,
    )
    assert empty[2] == "empty"

    loaded = service._process_node(
        _make_node("other", "Hello"),
        filters=filters,
        progress=progress,
        settings=settings,
        encoding=None,
    )
    assert loaded[2] == "loaded"

    skipped = service._process_node(
        _make_node("nope", "Hello"),
        filters=filters,
        progress=progress,
        settings=settings,
        encoding=None,
    )
    assert skipped[2] == "skipped"


def test_translate_nodes_async_requires_provider_settings(tmp_path):
    provider = DummyTranslationProvider()
    service = TranslationService(provider, provider_settings=None)
    filters = TranslationFilters(
        allowed_pattern=re.compile(r"^keep"),
        ignore_pattern=re.compile(r"^skip"),
    )
    settings = TranslationSettings(
        model="gpt-5-nano",
        target_language="Spanish",
        dry_run=False,
        count_tokens_enabled=False,
    )
    progress = TranslationProgress(
        done={},
        progress_file=tmp_path / "progress.json",
        progress_lock=Lock(),
    )
    summary = SummaryReporter("translate")

    with pytest.raises(ValueError):
        async def _run():
            return await service.translate_nodes_async(
                [],
                filters=filters,
                progress=progress,
                settings=settings,
                summary=summary,
            )

        anyio.run(_run)


def test_translate_nodes_async_dry_run_records_tuple(tmp_path):
    provider = DummyTranslationProvider()
    service = TranslationService(provider, provider_settings=_make_provider_settings())
    filters = TranslationFilters(
        allowed_pattern=re.compile(r"^keep"),
        ignore_pattern=re.compile(r"^skip"),
    )
    settings = TranslationSettings(
        model="gpt-5-nano",
        target_language="Spanish",
        dry_run=True,
        count_tokens_enabled=False,
    )
    progress = TranslationProgress(
        done={},
        progress_file=tmp_path / "progress.json",
        progress_lock=Lock(),
    )
    summary = SummaryReporter("translate")

    async def _run():
        return await service.translate_nodes_async(
            [_make_node("keep_one", "Hello")],
            filters=filters,
            progress=progress,
            settings=settings,
            summary=summary,
        )

    results = anyio.run(_run)

    assert isinstance(results[0][2], tuple)


def test_translate_nodes_async_failure_callback(monkeypatch, tmp_path):
    provider = DummyTranslationProvider()
    service = TranslationService(provider, provider_settings=_make_provider_settings())
    filters = TranslationFilters(
        allowed_pattern=re.compile(r"^keep"),
        ignore_pattern=re.compile(r"^skip"),
    )
    settings = TranslationSettings(
        model="gpt-5-nano",
        target_language="Spanish",
        dry_run=False,
        count_tokens_enabled=False,
    )
    progress = TranslationProgress(
        done={},
        progress_file=tmp_path / "progress.json",
        progress_lock=Lock(),
    )
    summary = SummaryReporter("translate")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_process_node", _boom)

    async def _run():
        return await service.translate_nodes_async(
            [_make_node("keep_fail", "Hello")],
            filters=filters,
            progress=progress,
            settings=settings,
            summary=summary,
        )

    results = anyio.run(_run)

    assert results
    assert "keep_fail" in summary.failures


class DummyResponse:
    def __init__(self, status_code: int, content: bytes = b"ok", text: str = "ok"):
        self.status_code = status_code
        self.content = content
        self.text = text


class DummyVoiceProvider:
    def __init__(self, responses: list[DummyResponse]):
        self._responses = responses
        self.calls = 0

    def build_headers(self, settings: ProviderSettings) -> dict[str, str]:
        return {"Authorization": f"Bearer {settings.api_key}"}

    def build_payload(self, text: str, *, settings: VoiceSettings) -> dict[str, str]:
        return {"text": text, "voice": settings.voice_name}

    async def send_request(self, *_args, **_kwargs) -> DummyResponse:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def test_voice_service_synthesize_once_success(tmp_path):
    provider = DummyVoiceProvider([DummyResponse(status_code=200, content=b"audio")])
    service = VoiceService(provider, provider_settings=_make_provider_settings())
    out_path = tmp_path / "out.mp3"

    async def _run():
        return await service.synthesize_once(
            client=object(),
            headers={"a": "b"},
            payload={"text": "hi"},
            out_path=out_path,
            max_elapsed_seconds=None,
        )

    result = anyio.run(_run)

    assert result == out_path
    assert out_path.read_bytes() == b"audio"


def test_voice_service_synthesize_once_http_error(tmp_path):
    provider = DummyVoiceProvider([DummyResponse(status_code=500, text="bad")])
    service = VoiceService(provider, provider_settings=_make_provider_settings())

    with pytest.raises(RuntimeError):
        async def _run():
            return await service.synthesize_once(
                client=object(),
                headers={"a": "b"},
                payload={"text": "hi"},
                out_path=tmp_path / "out.mp3",
                max_elapsed_seconds=None,
            )

        anyio.run(_run)


def test_voice_service_synthesize_once_timeout(monkeypatch, tmp_path):
    provider = DummyVoiceProvider([DummyResponse(status_code=200)])
    service = VoiceService(provider, provider_settings=_make_provider_settings())
    times = iter([0.0, 2.0])
    monkeypatch.setattr("apis.voice_api.time.perf_counter", lambda: next(times))

    with pytest.raises(TimeoutError):
        async def _run():
            return await service.synthesize_once(
                client=object(),
                headers={"a": "b"},
                payload={"text": "hi"},
                out_path=tmp_path / "out.mp3",
                max_elapsed_seconds=1.0,
            )

        anyio.run(_run)


def test_run_voice_async_success_and_skip(tmp_path):
    provider = DummyVoiceProvider([DummyResponse(status_code=200, content=b"ok")])
    settings = _make_provider_settings()
    service = VoiceService(provider, provider_settings=settings)
    voice_settings = VoiceSettings(
        model="octave",
        voice_name="ivan",
        provider="hume_ai",
        audio_format="mp3",
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )
    summary = SummaryReporter("voice")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def _run():
        return await service.run_voice_async(
            [("keep_one", "Hello")],
            output_dir=output_dir,
            settings=voice_settings,
            summary=summary,
            skipped_count=2,
        )

    results = anyio.run(_run)

    assert results == [("keep_one", "ok")]
    assert (output_dir / "keep_one.mp3").exists()
    assert summary.skipped == 2


def test_run_voice_async_failure_records_error(monkeypatch, tmp_path):
    provider = DummyVoiceProvider([DummyResponse(status_code=500, text="bad")])
    settings = _make_provider_settings()
    service = VoiceService(provider, provider_settings=settings)
    voice_settings = VoiceSettings(
        model="octave",
        voice_name="ivan",
        provider="hume_ai",
        audio_format="mp3",
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )
    summary = SummaryReporter("voice")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def _fake_run_with_progress(
        _name,
        _total,
        _runner,
        specs,
        _summary,
        *,
        success_cb=None,
        failure_cb=None,
        retry_cb=None,
    ):
        for spec in specs:
            if failure_cb:
                failure_cb(spec, RuntimeError("fail"))

    monkeypatch.setattr("apis.voice_api.run_with_progress", _fake_run_with_progress)

    async def _run():
        return await service.run_voice_async(
            [("keep_bad", "Hello")],
            output_dir=output_dir,
            settings=voice_settings,
            summary=summary,
            skipped_count=0,
        )

    results = anyio.run(_run)

    assert results == [("keep_bad", "error")]
    assert "keep_bad" in summary.failures


def test_run_voice_async_retries_and_logs(monkeypatch, tmp_path):
    provider = DummyVoiceProvider(
        [
            DummyResponse(status_code=500, text="fail"),
            DummyResponse(status_code=200, content=b"ok"),
        ]
    )
    settings = ProviderSettings(
        api_key="dummy",
        model="octave",
        rpm=1000,
        concurrency=1,
        retry=RetryConfig(attempts=2, backoff_base=0.0, backoff_max=0.0, jitter=False),
    )
    service = VoiceService(provider, provider_settings=settings)
    voice_settings = VoiceSettings(
        model="octave",
        voice_name="ivan",
        provider="hume_ai",
        audio_format="mp3",
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )
    summary = SummaryReporter("voice")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    calls: list[tuple[str, str, int]] = []

    def _fake_log_retry(command, task_id, attempt, total, sleep_for):
        calls.append((command, task_id, attempt))

    monkeypatch.setattr("apis.voice_api.log_retry", _fake_log_retry)

    async def _fake_run_with_progress(
        _name,
        _total,
        _runner,
        specs,
        _summary,
        *,
        success_cb=None,
        failure_cb=None,
        retry_cb=None,
    ):
        for spec in specs:
            if retry_cb:
                retry_cb(spec, 1, 0.0)
            if success_cb:
                success_cb(spec, output_dir / "keep_retry.mp3")

    monkeypatch.setattr("apis.voice_api.run_with_progress", _fake_run_with_progress)

    async def _run():
        return await service.run_voice_async(
            [("keep_retry", "Hello")],
            output_dir=output_dir,
            settings=voice_settings,
            summary=summary,
            skipped_count=0,
        )

    results = anyio.run(_run)

    assert results == [("keep_retry", "ok")]
    assert calls


def test_run_voice_async_requires_provider_settings(tmp_path):
    provider = DummyVoiceProvider([DummyResponse(status_code=200)])
    service = VoiceService(provider, provider_settings=None)
    voice_settings = VoiceSettings(
        model="octave",
        voice_name="ivan",
        provider="hume_ai",
        audio_format="mp3",
        split_utterances=True,
        octave_version="2",
        max_elapsed_seconds=None,
    )
    summary = SummaryReporter("voice")

    with pytest.raises(ValueError):
        async def _run():
            return await service.run_voice_async(
                [],
                output_dir=tmp_path,
                settings=voice_settings,
                summary=summary,
                skipped_count=0,
            )

        anyio.run(_run)
