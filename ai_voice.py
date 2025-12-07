#!/usr/bin/env python3
import json
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict

import requests
import toml

# ----------------------
# Configuration
# ----------------------
MODEL = "octave"  # API expects literal "octave"
VOICE_NAME = "ivan"
VOICE_PROVIDER = "HUME_AI"
AUDIO_FORMAT = "mp3"
SPLIT_UTTERANCES = True
NAME_REGEX = re.compile(r"^chp")
REQUEST_DELAY_SECONDS = 5.5  # 75 req/min
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]
STOP_AFTER = 0  # stop after this many API calls; 0 means no limit
MAX_WORKERS = 4  # number of parallel threads

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / ".cache/progress.json"
OUTPUT_DIR = BASE_DIR / "out/hume/"
CONFIG_PATH = BASE_DIR / "config.toml"
API_URL = "https://api.hume.ai/v0/tts/file"
LOG_LOCK = Lock()


def load_api_key() -> str:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config file: {CONFIG_PATH}. Populate hume_ai.api_key.")
    config = toml.load(CONFIG_PATH)
    api_key = config.get("hume_ai", {}).get("api_key", "")
    if not api_key:
        raise SystemExit(f"Set hume_ai.api_key in {CONFIG_PATH}.")
    return api_key


API_KEY = load_api_key()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_payload(text: str) -> Dict[str, Any]:
    """Construct request payload for a single utterance."""
    return {
        "model": MODEL,
        "format": {"type": AUDIO_FORMAT},
        "split_utterances": SPLIT_UTTERANCES,
        "version": "2",
        "utterances": [
            {
                "text": text,
                "voice": {"name": VOICE_NAME, "provider": VOICE_PROVIDER},
            }
        ],
    }


def send_request(headers: Dict[str, str], payload: Dict[str, Any]) -> requests.Response:
    """POST to Hume TTS endpoint."""
    return requests.post(API_URL, headers=headers, json=payload, timeout=120)


def main():
    ensure_api_key()
    log("starting ai_voice run")
    if not INPUT_FILE.exists():
        raise SystemExit(f"Progress file not found: {INPUT_FILE}. Run ai_transalte.py first.")
    progress = load_json(INPUT_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)
    headers = build_headers()
    rate_limiter = RateLimiter(REQUEST_DELAY_SECONDS)
    existing_outputs = get_existing_outputs(OUTPUT_DIR)
    log(f"found {len(existing_outputs)} existing outputs; skipping those keys")

    # Precompute worklist respecting STOP_AFTER to keep request count bounded.
    worklist: list[tuple[str, str]] = []
    log("building worklist")
    for key, text in progress.items():
        if STOP_AFTER and len(worklist) >= STOP_AFTER:
            log(f"stop_after reached at {STOP_AFTER} items")
            break
        if key in existing_outputs:
            log(f"processing {key} skip (already generated)")
            continue
        if not NAME_REGEX.match(key):
            log(f"processing {key} skip (regex)")
            continue
        out_path = OUTPUT_DIR / f"{key}.mp3"
        if out_path.exists():
            log(f"processing {key} skip (exists)")
            continue
        worklist.append((key, text))

    log(f"worklist size: {len(worklist)}")

    if not worklist:
        log("nothing to process")
        return

    # Fan out work items; rate limiter will serialize API pacing.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for key, text in worklist:
            futures.append(
                executor.submit(
                    handle_entry,
                    key=key,
                    text=text,
                    headers=headers,
                    rate_limiter=rate_limiter,
                )
            )
        for future in futures:
            future.result()
    log("ai_voice run complete")


def ensure_api_key() -> None:
    if not API_KEY:
        raise SystemExit(f"Set hume_ai.api_key in {CONFIG_PATH}.")


def build_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Hume-Api-Key": API_KEY,
    }


def handle_entry(
    *,
    key: str,
    text: str,
    headers: Dict[str, str],
    rate_limiter: "RateLimiter",
) -> None:
    """Process a single progress entry end-to-end (thread-safe)."""
    log(f"processing {key} start")
    out_path = OUTPUT_DIR / f"{key}.mp3"
    payload = build_payload(text)

    success, error_message = attempt_send(
        payload=payload,
        headers=headers,
        out_path=out_path,
        rate_limiter=rate_limiter,
    )

    if not success:
        log(f"processing {key} error: {error_message}")
    else:
        log(f"processing {key} done")


def get_existing_outputs(out_dir: Path) -> set[str]:
    """Return stems of existing MP3 files to avoid regenerating audio."""
    stems: set[str] = set()
    if not out_dir.exists():
        return stems
    for path in out_dir.glob("*.mp3"):
        stems.add(path.stem)
    return stems


def attempt_send(
    *,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    out_path: Path,
    rate_limiter: "RateLimiter",
) -> tuple[bool, str]:
    """Handle retries, backoff, and file write for one payload; obeys rate limit."""
    attempt = 0
    success = False
    error_message = ""

    while attempt < MAX_RETRIES and not success:
        log(f"processing {out_path.stem} attempt {attempt + 1}")
        rate_limiter.wait()
        attempt += 1
        try:
            response = send_request(headers, payload)
            if response.status_code == 200:
                out_path.write_bytes(response.content)
                success = True
            else:
                error_message = f"HTTP {response.status_code}: {response.text}"
        except requests.RequestException as exc:
            error_message = str(exc)

        if not success and attempt < MAX_RETRIES:
            log(f"processing {out_path.stem} retry after backoff {BACKOFF_SECONDS[attempt - 1]}s")
            time.sleep(BACKOFF_SECONDS[attempt - 1])

    return success, error_message


class RateLimiter:
    """Thread-safe rate limiter that enforces a minimum delay between requests."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = Lock()
        self._last: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = time.perf_counter()
            if self._last is not None:
                sleep_for = self.min_interval - (now - self._last)
                if sleep_for > 0:
                    log(f"rate limiter sleeping {sleep_for:.2f}s")
                    time.sleep(sleep_for)
                    now = time.perf_counter()
            self._last = now


def log(message: str) -> None:
    """Thread-safe logger."""
    with LOG_LOCK:
        print(message, flush=True)


if __name__ == "__main__":
    main()
