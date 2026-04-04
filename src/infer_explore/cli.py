"""CLI entry points for fetching model data."""

import sys


def fetch_aa() -> None:
    """Fetch data from Artificial Analysis."""
    from infer_explore.sources.artificial_analysis import fetch_and_save

    try:
        fetch_and_save()
        print("\nDone!")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_vantage() -> None:
    """Fetch data from Vantage."""
    from infer_explore.sources.vantage import fetch_and_save

    try:
        fetch_and_save()
        print("\nDone!")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_hf() -> None:
    """Fetch data from HuggingFace OpenEvals."""
    from infer_explore.sources.huggingface import fetch_and_save

    try:
        fetch_and_save()
        print("\nDone!")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_bifrost() -> None:
    """Fetch data from Bifrost."""
    from infer_explore.sources.bifrost import fetch_and_save

    try:
        fetch_and_save()
        print("\nDone!")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def merge_all() -> None:
    """Merge all data sources into a unified dataset."""
    from infer_explore.sources.merged import merge_and_save

    try:
        merge_and_save()
        print("\nDone!")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_all() -> None:
    """Fetch data from all sources."""
    from infer_explore.sources.artificial_analysis import (
        fetch_and_save as aa_fetch,
    )
    from infer_explore.sources.bifrost import fetch_and_save as bifrost_fetch
    from infer_explore.sources.huggingface import fetch_and_save as hf_fetch
    from infer_explore.sources.vantage import fetch_and_save as vantage_fetch

    sources = [
        ("Artificial Analysis", aa_fetch),
        ("Vantage", vantage_fetch),
        ("HuggingFace OpenEvals", hf_fetch),
        ("Bifrost", bifrost_fetch),
    ]

    failed = []
    for name, fetcher in sources:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        try:
            fetcher()
        except Exception as e:
            print(f"ERROR fetching {name}: {e}", file=sys.stderr)
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Completed with errors: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All sources fetched successfully!")
