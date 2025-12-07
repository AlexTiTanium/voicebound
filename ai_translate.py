import re
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from html import unescape
from openai import OpenAI
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import toml
import argparse
import sys

log_lock = Lock()
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.toml"


def load_api_key() -> str:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config file: {CONFIG_PATH}. Populate openai.api_key.")
    config = toml.load(CONFIG_PATH)
    api_key = config.get("openai", {}).get("api_key", "")
    if not api_key:
        raise SystemExit(f"Set openai.api_key in {CONFIG_PATH}.")
    return api_key


client = OpenAI(api_key=load_api_key())

# =======================
# Settings
# =======================
BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Translate Android strings.xml with OpenAI.")
parser.add_argument(
    "input_file",
    nargs="?",
    default=str(BASE_DIR / "strings.xml"),
    help="Path to the input strings.xml (default: %(default)s)",
)
args = parser.parse_args()

INPUT_FILE = Path(args.input_file)
OUTPUT_FILE = BASE_DIR / "out/values/strings.xml"
PROGRESS_FILE = BASE_DIR / ".cache/progress.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

DRY_RUN = False            # True = no translation, just analysis
COUNT_TOKENS = True        # count tokens before translation
MAX_WORKERS = 20            # number of parallel threads

MODEL = "gpt-5-nano"

# regex filters
TRANSLATE_NAME_REGEX = re.compile(r"^chp10_")
IGNORE_NAME_REGEX = re.compile(r"app_name")

# token encoder
encoding = tiktoken.get_encoding('o200k_base')

def count_tokens(text: str):
    return len(encoding.encode(text))


def log(message: str):
    """Thread-safe stdout logger."""
    with log_lock:
        print(message, flush=True)


if not DRY_RUN:
    log("[INIT] Dry run disabled — performing live translation with OpenAI client.")


# =======================
# Load progress
# =======================
if Path(PROGRESS_FILE).exists():
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        raw_done = json.load(f)
else:
    raw_done = {}

DONE = raw_done.copy()

progress_lock = Lock()

# descriptions no longer generated
DESCRIPTIONS: dict = {}


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


# =======================
# OpenAI Translation Call
# =======================
def translate_text(text):
    prompt = f"""
Translate the following text into Russian in a literary, artistic manner.
Do not add anything, do not modify structure, only translate the meaning:

{text}
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        log(f"[ERROR] OpenAI call failed: {e}")
        raise

    # OpenAI SDK returns a ChatCompletionMessage with attribute access
    return response.choices[0].message.content.strip()


# =======================
# Process one XML string
# =======================
def process_string(node):
    name = node.get("name")
    original = (node.text or "").strip()
    matches_regex = bool(TRANSLATE_NAME_REGEX.match(name))

    if IGNORE_NAME_REGEX.match(name):
        if not DRY_RUN:
            log(f"[IGNORE] {name} matched IGNORE_NAME_REGEX.")
        return name, None, "ignored"

    if not original:
        if not DRY_RUN:
            log(f"[EMPTY] {name} has no content; skipping.")
        return name, None, "empty"

    # count tokens
    orig_tokens = count_tokens(original)

    # already translated earlier
    if name in DONE:
        if not DRY_RUN:
            suffix = "" if matches_regex else " (bypassing TRANSLATE_NAME_REGEX)"
            log(f"[CACHE] {name} loaded from {PROGRESS_FILE}{suffix}.")
        translated = clean_text(DONE[name])
        return name, translated, "loaded"

    # skip translation when regex does not match, but still keep original text
    if not matches_regex:
        if not DRY_RUN:
            log(f"[SKIP] {name} does not match TRANSLATE_NAME_REGEX.")
        return name, clean_text(original), "skipped"

    # dry run mode → no translation
    if DRY_RUN:
        return name, None, ("dry-run", orig_tokens, original[:80])

    # actual translation
    log(f"[TRANSLATE] {name} — {orig_tokens} tokens; sending to OpenAI.")
    try:
        translated = translate_text(original)
    except Exception as e:
        log(f"[ERROR] {name} translation failed: {e}")
        return name, None, ("error", str(e))

    log(f"[RECEIVED] {name} translation received; persisting.")
    translated = clean_text(translated)

    # save progress thread-safely
    with progress_lock:
        DONE[name] = translated
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(DONE, f, ensure_ascii=False, indent=2)

    log(f"[SAVED] {name} progress updated in {PROGRESS_FILE}.")

    return name, translated, "translated"


# =======================
# Main routine
# =======================
if not INPUT_FILE.exists():
    raise SystemExit(f"Input file not found: {INPUT_FILE}")

tree = ET.parse(INPUT_FILE)
root = tree.getroot()

tasks = []
results = []

# collect all translatable nodes
for node in root.findall("string"):
    name = node.get("name")
    tasks.append(node)

# multithreading
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {executor.submit(process_string, n): n for n in tasks}

    for future in as_completed(future_map):
        try:
            name, result, status = future.result()
        except Exception as e:
            # unexpected error bubble-up safeguard
            name = future_map[future].get("name")
            log(f"[ERROR] Unhandled exception for {name}: {e}")
            results.append((name, None, ("error", str(e))))
            continue

        results.append((name, result, status))


# =======================
# Apply translation to XML
# =======================
if not DRY_RUN:
    for name, text, status in results:
        if status in ("translated", "loaded", "skipped"):
            text = clean_text(text)
            for node in root.findall("string"):
                if node.get("name") == name and text is not None:
                    node.text = text

    log(f"[WRITE] Applying translations to {OUTPUT_FILE}.")
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    log("=== DONE ===")
    log(f"Output saved to: {OUTPUT_FILE}")

else:
    # dry run summary
    print("\n=== DRY RUN REPORT ===")
    total_tokens = 0
    count = 0

    for name, data, status in results:
        if isinstance(status, tuple) and status[0] == "dry-run":
            _, tokens, preview = status
            total_tokens += tokens
            count += 1
            print(f"[DRY] {name}: {tokens} tokens → '{preview}...'")

    print("\n=== SUMMARY ===")
    print("Strings to be translated:", count)
    print("Total tokens:", total_tokens)
    print("Estimated cost:")
    print(f"- Input: {total_tokens} tokens (~{total_tokens/1_000_000 * 0.005:.4f}$)")
    print(f"- Output estimate: {total_tokens} tokens (~{total_tokens/1_000_000 * 0.40:.4f}$)")
    print("No translation performed (dry-run mode).")
