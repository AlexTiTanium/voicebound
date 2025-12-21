import functools
import re
from html import unescape
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Optional, Tuple, TypedDict

import anyio
import tiktoken
import typer
from loguru import logger
from openai import OpenAI

from command_context import CommandContext, make_command_context, run_with_progress
from command_utils import (
    ProviderSettings,
    build_runner,
    build_task_specs,
    load_progress,
    load_strings,
    persist_progress,
)
from summary_reporter import SummaryReporter
from task_runner import TaskHooks, TaskSpec
from utils import (
    PROJECT_ROOT,
    compile_regex,
    ensure_directory,
    get_config_value,
    resolve_path,
)


class Summary(TypedDict):
    translated: int
    skipped: int
    loaded: int
    ignored: int
    empty: int
    errors: list[str]


def clean_text(text: str | None) -> str | None:
    """Convert escaped markers to real characters; unescape XML entities."""
    if text is None:
        return None
    text = (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )
    return unescape(text)


def translate_text(client: Any, text: str, model: str, target_language: str) -> str:
    """Call the OpenAI chat completion API to translate text into the target language."""
    prompt = f"""
Translate the following text into {target_language} in a literary, artistic manner.
Do not add anything, do not modify structure, only translate the meaning:

{text}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def count_tokens(encoding, text: str) -> int:
    """Return the token count for the given text using the provided encoding."""
    return len(encoding.encode(text))


def process_string(
    node,
    *,
    translate_pattern: re.Pattern[str],
    ignore_pattern: re.Pattern[str],
    done: dict,
    progress_lock: Lock,
    client: OpenAI,
    progress_file: Path,
    model: str,
    dry_run: bool,
    encoding,
    target_language: str,
) -> Tuple[str | None, str | None, str | tuple]:
    name = node.get("name") or ""
    original = (node.text or "").strip()
    matches_regex = bool(translate_pattern.match(name))

    if ignore_pattern.match(name):
        logger.debug(f"[SKIP] {name} matched ignore regex.")
        return name, None, "ignored"

    if not original:
        logger.debug(f"[SKIP] {name} has no content.")
        return name, None, "empty"

    orig_tokens = count_tokens(encoding, original) if encoding else 0

    if name in done:
        suffix = "" if matches_regex else " (bypassing translate regex)"
        logger.info(f"[CACHE] {name} loaded{suffix}.")
        translated = clean_text(done[name])
        return name, translated, "loaded"

    if not matches_regex:
        logger.debug(f"[SKIP] {name} does not match translate regex.")
        return name, clean_text(original), "skipped"

    if dry_run:
        return name, None, ("dry-run", orig_tokens, original[:80])

    if orig_tokens:
        logger.info(f"[TRANSLATE] {name} — {orig_tokens} tokens.")
    else:
        logger.info(f"[TRANSLATE] {name}.")

    try:
        translated = translate_text(client, original, model, target_language)
    except Exception as exc:  # pragma: no cover - API failure surface
        logger.error(f"[ERROR] {name} translation failed: {exc}")
        return name, None, ("error", str(exc))
    translated = clean_text(translated)

    with progress_lock:
        done[name] = translated
        persist_progress(progress_file, done)

    logger.info(f"[SAVED] {name} progress updated in {progress_file}.")
    return name, translated, "translated"


def apply_translations(
    root, results: Iterable[tuple[str | None, str | None, str | tuple]]
) -> None:
    """Apply translated text back to the XML tree for eligible entries."""
    for name, text, status in results:
        if status in ("translated", "loaded", "skipped"):
            text = clean_text(text)
            for node in root.findall("string"):
                if node.get("name") == name and text is not None:
                    node.text = text


def translate_strings(
    input_file: Path | None = None,
    output_file: Path | None = None,
    progress_file: Path | None = None,
    *,
    allowed_regex: str | None = None,
    ignore_regex: str | None = None,
    dry_run: bool | None = None,
    max_workers: int | None = None,
    model: str | None = None,
    count_tokens_enabled: bool | None = None,
    target_language: str | None = None,
    config_path: Path | None = PROJECT_ROOT / "config.toml",
    log_level: str | None = None,
    color: bool = True,
) -> None:
    """Translate strings.xml using OpenAI according to config and CLI overrides."""
    ctx: CommandContext = make_command_context(
        config_path=config_path,
        provider_key="openai",
        default_model="gpt-5-nano",
        default_rpm=60,
        concurrency_override=max_workers,
        log_level=log_level,
        color=color,
    )
    api_key = ctx.provider.api_key
    model = model or ctx.provider.model
    translate_cfg = ctx.config.get("translate", {})
    provider = get_config_value(
        ctx.config, "translate", "provider", required=False, default="openai"
    )
    if str(provider).lower() != "openai":
        logger.warning(
            f"[TRANSLATE] Provider '{provider}' is not recognized; defaulting to OpenAI client."
        )

    input_file = resolve_path(input_file or translate_cfg.get("input_file", "strings.xml"))
    output_file = resolve_path(
        output_file or translate_cfg.get("output_file", "out/values/strings.xml")
    )
    progress_file = resolve_path(
        progress_file or translate_cfg.get("progress_file", ".cache/progress.json")
    )
    allowed_regex = allowed_regex or translate_cfg.get("allowed_regex", r"^chp10_")
    ignore_regex = ignore_regex or translate_cfg.get("ignore_regex", r"app_name")
    dry_run = translate_cfg.get("dry_run", False) if dry_run is None else dry_run
    max_workers = translate_cfg.get("max_workers", 20) if max_workers is None else max_workers
    count_tokens_enabled = (
        translate_cfg.get("count_tokens_enabled", True)
        if count_tokens_enabled is None
        else count_tokens_enabled
    )
    target_language = target_language or translate_cfg.get("target_language", "Russian")

    translate_pattern = compile_regex(allowed_regex, label="allowed")
    ignore_pattern = compile_regex(ignore_regex, label="ignore")

    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")

    ensure_directory(output_file.parent)
    ensure_directory(progress_file.parent)

    tree, root = load_strings(input_file)

    done = load_progress(progress_file, default={}) or {}
    progress_lock = Lock()
    tasks = list(root.findall("string"))
    pre_results: list[tuple[str | None, str | None, str | tuple]] = []
    translate_nodes: list = []
    for node in tasks:
        name = node.get("name") or ""
        original = (node.text or "").strip()
        matches_regex = bool(translate_pattern.match(name))

        if ignore_pattern.match(name):
            pre_results.append((name, None, "ignored"))
            continue

        if not original:
            pre_results.append((name, None, "empty"))
            continue

        if name in done:
            pre_results.append((name, clean_text(done[name]), "loaded"))
            continue

        if not matches_regex:
            pre_results.append((name, clean_text(original), "skipped"))
            continue

        translate_nodes.append(node)

    if not dry_run:
        logger.info(f"[TRANSLATE] Translating {len(translate_nodes)} strings using {model}.")
    else:
        logger.info(f"[TRANSLATE] Dry run enabled for {len(translate_nodes)} strings.")

    encoding = tiktoken.get_encoding("o200k_base") if count_tokens_enabled else None
    client = OpenAI(api_key=api_key)
    try:
        summary = SummaryReporter("translate")
        for name, _, status in pre_results:
            if isinstance(status, str):
                summary.record_translation(status, name)
        results = anyio.run(
            _run_translate_async,
            translate_nodes,
            translate_pattern,
            ignore_pattern,
            done,
            progress_lock,
            client,
            progress_file,
            model,
            dry_run,
            encoding,
            target_language,
            ctx.provider,
            summary,
        )
        results = pre_results + results
    except KeyboardInterrupt:
        logger.warning("[TRANSLATE] Interrupted by user.")
        raise SystemExit(130)

    if dry_run:
        _print_dry_run(results)
        return

    apply_translations(root, results)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


async def _run_translate_async(
    tasks: list,
    translate_pattern: re.Pattern[str],
    ignore_pattern: re.Pattern[str],
    done: dict,
    progress_lock: Lock,
    client: Any,
    progress_file: Path,
    model: str,
    dry_run: bool,
    encoding: Any,
    target_language: str,
    provider_settings: ProviderSettings,
    summary: SummaryReporter,
) -> list[tuple[str | None, str | None, str | tuple]]:
    """Drive TaskRunner for translation tasks."""
    results: list[tuple[str | None, str | None, str | tuple]] = []
    runner = build_runner(
        "translate",
        provider_settings,
        TaskHooks(),
    )
    work_items: list[tuple[str, Callable[..., Any]]] = []
    for idx, node in enumerate(tasks):
        task_name = node.get("name") or f"string-{idx}"
        task_fn = functools.partial(
            process_string,
            node,
            translate_pattern=translate_pattern,
            ignore_pattern=ignore_pattern,
            done=done,
            progress_lock=progress_lock,
            client=client,
            progress_file=progress_file,
            model=model,
            dry_run=dry_run,
            encoding=encoding,
            target_language=target_language,
        )

        async def coro(fn=task_fn):
            return await anyio.to_thread.run_sync(fn)

        work_items.append((task_name, coro))

    specs = build_task_specs(work_items)
    def success_cb(spec: TaskSpec, result):
        results.append(result)
        status = result[2]
        if isinstance(status, tuple):
            summary.record_translation(status[0], result[0])
        else:
            summary.record_translation(status, result[0])

    def failure_cb(spec: TaskSpec, exc: BaseException):
        results.append((spec.task_id, None, ("error", str(exc))))
        summary.record_translation("error", spec.task_id)

    await run_with_progress(
        "translate",
        len(specs),
        runner,
        specs,
        summary,
        success_cb=success_cb,
        failure_cb=failure_cb,
    )
    summary.log_translation(str(progress_file))
    return results


def _print_dry_run(results: Iterable[tuple[str | None, str | None, str | tuple]]) -> None:
    """Print/log a dry-run summary showing counts and token estimates."""
    total_tokens = 0
    count = 0

    for name, data, status in results:
        if isinstance(status, tuple) and status[0] == "dry-run":
            _, tokens, preview = status
            total_tokens += tokens
            count += 1
            logger.info(f"[DRY] {name}: {tokens} tokens → '{preview}...'")

    logger.info("=== SUMMARY ===")
    logger.info(f"Strings to be translated: {count}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Estimated input cost (~${total_tokens/1_000_000 * 0.005:.4f})")
    logger.info(f"Estimated output cost (~${total_tokens/1_000_000 * 0.40:.4f})")
    logger.info("No translation performed (dry-run mode).")


def typer_command(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, help="Path to the input strings.xml."),
    output_file: Optional[Path] = typer.Option(None, help="Path to write translated XML."),
    allowed_regex: Optional[str] = typer.Option(
        None, help="Translate only entries matching this regex."
    ),
    ignore_regex: Optional[str] = typer.Option(None, help="Ignore entries matching this regex."),
    dry_run: Optional[bool] = typer.Option(None, help="Dry run (no translation calls)."),
    max_workers: Optional[int] = typer.Option(None, help="Parallel workers."),
    model: Optional[str] = typer.Option(None, help="OpenAI model to use."),
    target_language: Optional[str] = typer.Option(None, help="Target language to translate into."),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.toml."),
) -> None:
    """Typer CLI wrapper for translate_strings."""
    obj = ctx.ensure_object(dict)
    cfg_raw = config_path or obj.get("config_path")
    cfg_path = Path(cfg_raw) if cfg_raw else None
    log_level = obj.get("log_level")
    color = obj.get("color", True)
    translate_strings(
        input_file=input_file,
        output_file=output_file,
        allowed_regex=allowed_regex,
        ignore_regex=ignore_regex,
        dry_run=dry_run,
        max_workers=max_workers,
        model=model,
        target_language=target_language,
        config_path=cfg_path,
        log_level=log_level,
        color=color,
    )


def _summarize_results(results: Iterable[tuple[str | None, str | None, str | tuple]]) -> Summary:
    summary: Summary = {
        "translated": 0,
        "skipped": 0,
        "loaded": 0,
        "ignored": 0,
        "empty": 0,
        "errors": [],
    }
    for name, _text, status in results:
        if status == "translated":
            summary["translated"] += 1
        elif status == "skipped":
            summary["skipped"] += 1
        elif status == "loaded":
            summary["loaded"] += 1
        elif status == "ignored":
            summary["ignored"] += 1
        elif status == "empty":
            summary["empty"] += 1
        elif isinstance(status, tuple) and status[0] == "error":
            if name:
                summary["errors"].append(name)
    return summary


if __name__ == "__main__":
    typer.run(typer_command)
