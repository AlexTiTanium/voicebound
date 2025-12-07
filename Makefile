PY = python
PIP = $(PY) -m pip
VENV ?= ./venv
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff

.PHONY: install lint format type test translate voice

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

lint:
	$(RUFF) check .

type:
	$(VENV)/bin/mypy src tests

format:
	$(RUFF) format .

test:
	$(PYTEST) --cov=src --cov-report=term-missing

translate:
	$(VENV)/bin/python main.py translate --help

voice:
	$(VENV)/bin/python main.py voice --help
