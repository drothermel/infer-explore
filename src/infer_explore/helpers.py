"""Shared helpers for data fetching and saving."""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def get_project_root() -> Path:
    """Return the project root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve().parent
    # Walk up until we find pyproject.toml
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent  # fallback: src/infer_explore -> root


def get_data_dir() -> Path:
    """Return the data/ directory, creating it if needed."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_env() -> None:
    """Load .env from the project root if it exists."""
    env_path = get_project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def get_env_key(name: str, required: bool = True) -> str | None:
    """Get an environment variable, optionally raising if missing."""
    load_env()
    val = os.environ.get(name, "").strip()
    if not val and required:
        raise EnvironmentError(
            f"{name} not set. Copy .env.example to .env and fill in your key."
        )
    return val or None


def save_json(data: dict | list, path: Path, source_url: str | None = None) -> None:
    """Save data as pretty-printed JSON with metadata."""
    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_url:
        output["source"] = source_url
    if isinstance(data, list):
        output["count"] = len(data)
        output["models"] = data
    else:
        output.update(data)

    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Saved JSON → {path.name} ({path.stat().st_size / 1024:.0f}K)")


def save_csv(
    rows: list[dict],
    fieldnames: list[str],
    path: Path,
) -> None:
    """Save a list of dicts as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV  → {path.name} ({len(rows)} rows)")


def print_summary(
    source_name: str,
    total: int,
    **counts: int,
) -> None:
    """Print a quick summary of fetched data."""
    print(f"\n  {source_name} summary:")
    print(f"    Total models: {total}")
    for label, n in counts.items():
        print(f"    {label}: {n}")
