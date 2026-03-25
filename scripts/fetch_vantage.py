#!/usr/bin/env python3
"""Fetch model data from the Vantage models site.

This is a thin wrapper around infer_explore.sources.vantage.
See `uv run fetch-vantage` for the installed entry point.

Usage:
    uv run python scripts/fetch_vantage.py
"""

from infer_explore.cli import fetch_vantage

if __name__ == "__main__":
    fetch_vantage()
