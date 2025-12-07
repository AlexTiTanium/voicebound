#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PY="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"

echo "Checking Python..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN is not installed."
  exit 1
fi

create_venv() {
  echo "Creating virtual environment..."
  rm -rf "$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
}

if [ ! -d "$VENV" ]; then
  create_venv
elif [ -f "$VENV_PIP" ]; then
  PIP_SHEBANG="$(head -n 1 "$VENV_PIP" 2>/dev/null || true)"
  if [[ "$PIP_SHEBANG" == \#!* ]]; then
    PIP_INTERPRETER="${PIP_SHEBANG:2}"
    if [ ! -x "$PIP_INTERPRETER" ]; then
      echo "Existing venv points to missing interpreter ($PIP_INTERPRETER); recreating..."
      create_venv
    fi
  fi
fi

if [ ! -x "$VENV_PY" ]; then
  echo "Error: virtualenv missing python at $VENV_PY"
  exit 1
fi

echo "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip

echo "Installing dependencies..."
"$VENV_PIP" install -r "$ROOT/requirements.txt"

echo "All dependencies installed successfully."
echo "Use '$VENV_PY script.py' to run without activating."

echo "Activating virtual environment..."

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "Virtual environment active. Type 'exit' to leave this shell."
exec "${SHELL:-/bin/bash}"
