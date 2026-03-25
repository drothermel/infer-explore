#!/usr/bin/env python3
"""Fetch model data from the HuggingFace OpenEvals leaderboard dataset.

This is a thin wrapper around infer_explore.sources.huggingface.
See `uv run fetch-hf` for the installed entry point.

Usage:
    uv run python scripts/fetch_huggingface.py
"""

from infer_explore.cli import fetch_hf

if __name__ == "__main__":
    fetch_hf()
