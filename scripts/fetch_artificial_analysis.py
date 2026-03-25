#!/usr/bin/env python3
"""Fetch model data from the Artificial Analysis free API.

This is a thin wrapper around infer_explore.sources.artificial_analysis.
See `uv run fetch-aa` for the installed entry point.

Usage:
    uv run python scripts/fetch_artificial_analysis.py
"""

from infer_explore.cli import fetch_aa

if __name__ == "__main__":
    fetch_aa()
