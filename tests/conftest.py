"""Shared test setup."""

import sys
from pathlib import Path

# Make ``tests/synthetic.py`` importable as ``synthetic`` from every test module.
sys.path.insert(0, str(Path(__file__).parent))
