#!/usr/bin/env python3
"""Fetch model data from all sources (Artificial Analysis, Vantage, HuggingFace).

This is a thin wrapper around infer_explore.cli.fetch_all.
See `uv run fetch-all` for the installed entry point.

Usage:
    uv run python scripts/fetch_all.py
"""

from infer_explore.cli import fetch_all

if __name__ == "__main__":
    fetch_all()
