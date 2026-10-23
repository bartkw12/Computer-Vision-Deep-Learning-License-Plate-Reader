.PHONY: help setup lint format test test-fast clean

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## create .venv and install the project with the dev and OCR extras
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,ocr]"

lint:  ## check style and likely bugs
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## apply formatting and safe lint fixes
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## run every test (model tests download weights on first use)
	$(BIN)/pytest

test-fast:  ## run the tests that need no model weights
	$(BIN)/pytest -m "not models"

clean:  ## remove caches and build leftovers
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
