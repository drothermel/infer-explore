#!/usr/bin/env python3
"""
Fetch model data from the Artificial Analysis free API.

Pulls benchmark scores, pricing, speed metrics, and release dates for ~450 LLMs.
Saves raw JSON and a flattened summary CSV to ../data/.

Usage:
    python scripts/fetch_artificial_analysis.py

Requires:
    - ARTIFICIAL_ANALYSIS_API_KEY in .env (or as an environment variable)
    - pip install requests python-dotenv  (both in requirements.txt)

API docs: https://artificialanalysis.ai/api-reference#models-endpoint
Rate limit: 1,000 requests/day
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Evaluation keys we care about (in display order)
EVAL_KEYS = [
    "artificial_analysis_intelligence_index",
    "artificial_analysis_coding_index",
    "artificial_analysis_math_index",
    "mmlu_pro",
    "gpqa",
    "hle",
    "livecodebench",
    "scicode",
    "math_500",
    "aime",
    "aime_25",
    "ifbench",
    "lcr",
    "terminalbench_hard",
    "tau2",
]

# Friendly labels for CSV header
EVAL_LABELS = {
    "artificial_analysis_intelligence_index": "AA Intelligence Index",
    "artificial_analysis_coding_index": "AA Coding Index",
    "artificial_analysis_math_index": "AA Math Index",
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA",
    "hle": "HLE",
    "livecodebench": "LiveCodeBench",
    "scicode": "SciCode",
    "math_500": "MATH-500",
    "aime": "AIME",
    "aime_25": "AIME 2025",
    "ifbench": "IFBench",
    "lcr": "LCR",
    "terminalbench_hard": "TerminalBench Hard",
    "tau2": "TAU-2",
}


def load_api_key() -> str:
    """Load the API key from .env or environment."""
    # Try loading .env from the project root
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "").strip()
    if not key:
        print("ERROR: ARTIFICIAL_ANALYSIS_API_KEY not set.", file=sys.stderr)
        print("  Copy .env.example to .env and fill in your key.", file=sys.stderr)
        sys.exit(1)
    return key


def fetch_models(api_key: str) -> list[dict]:
    """Fetch model data from the Artificial Analysis API."""
    print(f"Fetching models from {API_URL} ...")
    resp = requests.get(API_URL, headers={"x-api-key": api_key}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != 200:
        print(f"WARNING: API returned status {payload.get('status')}", file=sys.stderr)

    models = payload.get("data", [])
    print(f"  Received {len(models)} models")
    return models


def save_raw_json(models: list[dict], path: Path):
    """Save raw API response data as pretty-printed JSON."""
    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://artificialanalysis.ai/api/v2/data/llms/models",
        "count": len(models),
        "models": models,
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Saved raw JSON → {path.relative_to(PROJECT_ROOT)}")


def save_summary_csv(models: list[dict], path: Path):
    """Save a flattened CSV with one row per model."""
    fieldnames = [
        "name",
        "slug",
        "creator",
        "release_date",
        "input_price_per_1m",
        "output_price_per_1m",
        "blended_price_per_1m",
        "output_tokens_per_sec",
        "time_to_first_token_s",
        *[EVAL_LABELS.get(k, k) for k in EVAL_KEYS],
        "aa_id",
    ]

    rows = []
    for m in models:
        evals = m.get("evaluations") or {}
        pricing = m.get("pricing") or {}
        creator = (m.get("model_creator") or {}).get("name", "")

        row = {
            "name": m.get("name", ""),
            "slug": m.get("slug", ""),
            "creator": creator,
            "release_date": m.get("release_date", ""),
            "input_price_per_1m": pricing.get("price_1m_input_tokens"),
            "output_price_per_1m": pricing.get("price_1m_output_tokens"),
            "blended_price_per_1m": pricing.get("price_1m_blended_3_to_1"),
            "output_tokens_per_sec": m.get("median_output_tokens_per_second"),
            "time_to_first_token_s": m.get("median_time_to_first_token_seconds"),
            "aa_id": m.get("id", ""),
        }

        # Flatten evaluation scores
        for key in EVAL_KEYS:
            label = EVAL_LABELS.get(key, key)
            row[label] = evals.get(key)

        rows.append(row)

    # Sort by intelligence index descending (nulls last)
    rows.sort(
        key=lambda r: (
            r.get("AA Intelligence Index") is not None,
            r.get("AA Intelligence Index") or 0,
        ),
        reverse=True,
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved summary CSV → {path.relative_to(PROJECT_ROOT)}")


def print_summary(models: list[dict]):
    """Print a quick summary of the fetched data."""
    with_evals = sum(
        1
        for m in models
        if m.get("evaluations")
        and any(v is not None for v in m["evaluations"].values())
    )
    with_pricing = sum(1 for m in models if m.get("pricing"))
    with_speed = sum(1 for m in models if m.get("median_output_tokens_per_second"))
    with_date = sum(1 for m in models if m.get("release_date"))

    creators = {}
    for m in models:
        c = (m.get("model_creator") or {}).get("name", "Unknown")
        creators[c] = creators.get(c, 0) + 1

    print(f"\n  Summary:")
    print(f"    Total models:       {len(models)}")
    print(f"    With evaluations:   {with_evals}")
    print(f"    With pricing:       {with_pricing}")
    print(f"    With speed data:    {with_speed}")
    print(f"    With release date:  {with_date}")
    print(f"    Unique creators:    {len(creators)}")
    print(f"    Top creators:       ", end="")
    for name, count in sorted(creators.items(), key=lambda x: -x[1])[:8]:
        print(f"{name} ({count}), ", end="")
    print("...")


def main():
    api_key = load_api_key()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    models = fetch_models(api_key)

    # Save outputs
    save_raw_json(models, DATA_DIR / "artificial_analysis_models.json")
    save_summary_csv(models, DATA_DIR / "artificial_analysis_models.csv")

    print_summary(models)
    print("\nDone!")


if __name__ == "__main__":
    main()
