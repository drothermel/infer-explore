#!/usr/bin/env python3
"""Merge all data sources into a unified dataset.

This is a thin wrapper around infer_explore.sources.merged.
See `uv run merge-all` for the installed entry point.

Usage:
    uv run python scripts/merge_all.py
"""

from infer_explore.cli import merge_all

if __name__ == "__main__":
    merge_all()
