.PHONY: help setup lint format test test-fast figures readme readme-check demo clean

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

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

figures:  ## draw the README figures from results/ into assets/
	$(BIN)/python scripts/make_figures.py

readme:  ## regenerate the README's result tables from results/
	$(BIN)/python scripts/render_readme_tables.py

readme-check:  ## fail if the README disagrees with results/
	$(BIN)/python scripts/render_readme_tables.py --check

demo:  ## read the plates in VIDEO and render the annotated video, e.g. make demo VIDEO=data/videos/clip.mp4
	@test -n "$(VIDEO)" || { echo "usage: make demo VIDEO=path/to/video.mp4"; exit 2; }
	$(BIN)/alpr run "$(VIDEO)" --render

clean:  ## remove caches and build leftovers
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
