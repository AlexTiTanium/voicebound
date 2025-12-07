from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from threading import Lock
from typing import Iterable, Tuple
from xml.etree import ElementTree as ET

import tiktoken
import typer
from loguru import logger
from openai import OpenAI

from voicebound.utils import (
    PROJECT_ROOT,
    configure_logging,
    get_config_value,
    ensure_directory,
    load_config,
    load_json,
    resolve_path,
    write_json,
)


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


def translate_text(client: OpenAI, text: str, model: str) -> str:
    prompt = f"""
Translate the following text into Russian in a literary, artistic manner.
Do not add anything, do not modify structure, only translate the meaning:

{text}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def count_tokens(encoding, text: str) -> int:
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
        translated = translate_text(client, original, model)
    except Exception as exc:  # pragma: no cover - API failure surface
        logger.error(f"[ERROR] {name} translation failed: {exc}")
        return name, None, ("error", str(exc))
    translated = clean_text(translated)

    with progress_lock:
        done[name] = translated
        write_json(progress_file, done)

    logger.info(f"[SAVED] {name} progress updated in {progress_file}.")
    return name, translated, "translated"


def apply_translations(root, results: Iterable[tuple[str | None, str | None, str]]) -> None:
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
    translate_regex: str | None = None,
    ignore_regex: str | None = None,
    dry_run: bool | None = None,
    max_workers: int | None = None,
    model: str | None = None,
    count_tokens_enabled: bool | None = None,
    config_path: Path = PROJECT_ROOT / "config.toml",
) -> None:
    configure_logging()
    config = load_config(config_path)
    api_key = get_config_value(config, "openai", "api_key")
    translate_cfg = config.get("translate", {})

    input_file = resolve_path(input_file or translate_cfg.get("input_file", "strings.xml"))
    output_file = resolve_path(output_file or translate_cfg.get("output_file", "out/values/strings.xml"))
    progress_file = resolve_path(progress_file or translate_cfg.get("progress_file", ".cache/progress.json"))
    translate_regex = translate_regex or translate_cfg.get("translate_regex", r"^chp10_")
    ignore_regex = ignore_regex or translate_cfg.get("ignore_regex", r"app_name")
    dry_run = translate_cfg.get("dry_run", False) if dry_run is None else dry_run
    max_workers = translate_cfg.get("max_workers", 20) if max_workers is None else max_workers
    model = model or translate_cfg.get("model", "gpt-5-nano")
    count_tokens_enabled = (
        translate_cfg.get("count_tokens_enabled", True) if count_tokens_enabled is None else count_tokens_enabled
    )

    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")

    ensure_directory(output_file.parent)
    ensure_directory(progress_file.parent)

    tree = ET.parse(input_file)
    root = tree.getroot()

    done = load_json(progress_file, default={}) or {}
    progress_lock = Lock()
    tasks = list(root.findall("string"))

    if not dry_run:
        logger.info(f"[INIT] Translating {len(tasks)} strings using {model}.")
    else:
        logger.info(f"[INIT] Dry run enabled for {len(tasks)} strings.")

    encoding = tiktoken.get_encoding("o200k_base") if count_tokens_enabled else None
    translate_pattern = re.compile(translate_regex)
    ignore_pattern = re.compile(ignore_regex)
    client = OpenAI(api_key=api_key)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
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
            ): node
            for node in tasks
        }

        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - safeguard
                name = future_map[future].get("name")
                logger.error(f"[ERROR] Unhandled exception for {name}: {exc}")
                results.append((name, None, ("error", str(exc))))

    if dry_run:
        _print_dry_run(results)
        return

    apply_translations(root, results)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    logger.success(f"[WRITE] Output saved to: {output_file}")


def _print_dry_run(results: Iterable[tuple[str | None, str | None, str | tuple]]) -> None:
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
    input_file: Path | None = typer.Option(None, help="Path to the input strings.xml."),
    output_file: Path | None = typer.Option(None, help="Path to write translated XML."),
    progress_file: Path | None = typer.Option(None, help="Path to progress cache JSON."),
    translate_regex: str | None = typer.Option(None, help="Translate only entries matching this regex."),
    ignore_regex: str | None = typer.Option(None, help="Ignore entries matching this regex."),
    dry_run: bool | None = typer.Option(None, help="Dry run (no translation calls)."),
    max_workers: int | None = typer.Option(None, help="Parallel workers."),
    model: str | None = typer.Option(None, help="OpenAI model to use."),
    count_tokens_enabled: bool | None = typer.Option(None, help="Count tokens before translation."),
    config_path: Path = typer.Option(PROJECT_ROOT / "config.toml", help="Path to config.toml."),
) -> None:
    translate_strings(
        input_file=input_file,
        output_file=output_file,
        progress_file=progress_file,
        translate_regex=translate_regex,
        ignore_regex=ignore_regex,
        dry_run=dry_run,
        max_workers=max_workers,
        model=model,
        count_tokens_enabled=count_tokens_enabled,
        config_path=config_path,
    )


if __name__ == "__main__":
    typer.run(typer_command)
