#!/usr/bin/env python3
"""Fetch model data from the Bifrost datasheet.

This is a thin wrapper around infer_explore.sources.bifrost.
See `uv run fetch-bifrost` for the installed entry point.

Usage:
    uv run python scripts/fetch_bifrost.py
"""

from infer_explore.cli import fetch_bifrost

if __name__ == "__main__":
    fetch_bifrost()
