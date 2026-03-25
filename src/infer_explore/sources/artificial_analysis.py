"""Fetch model data from the Artificial Analysis free API.

Pulls benchmark scores, pricing, speed metrics, and release dates for ~450 LLMs.

API docs: https://artificialanalysis.ai/api-reference#models-endpoint
Rate limit: 1,000 requests/day (free tier)
"""

import requests

from infer_explore.helpers import (
    get_data_dir,
    get_env_key,
    print_summary,
    save_csv,
    save_json,
)

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

# Evaluation keys in display order
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


def fetch() -> list[dict]:
    """Fetch model data from the Artificial Analysis API."""
    api_key = get_env_key("ARTIFICIAL_ANALYSIS_API_KEY")
    print(f"Fetching models from Artificial Analysis ...")
    resp = requests.get(API_URL, headers={"x-api-key": api_key}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    models = payload.get("data", [])
    print(f"  Received {len(models)} models")
    return models


def _flatten_row(m: dict) -> dict:
    """Flatten one API model record into a CSV-friendly dict."""
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
    for key in EVAL_KEYS:
        row[EVAL_LABELS.get(key, key)] = evals.get(key)
    return row


CSV_FIELDS = [
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


def fetch_and_save() -> list[dict]:
    """Fetch, save raw JSON + flattened CSV, print summary. Returns raw models."""
    models = fetch()
    data_dir = get_data_dir()

    # Save raw JSON
    save_json(models, data_dir / "artificial_analysis_models.json", source_url=API_URL)

    # Flatten and sort by intelligence index (nulls last)
    rows = [_flatten_row(m) for m in models]
    rows.sort(
        key=lambda r: (
            r.get("AA Intelligence Index") is not None,
            r.get("AA Intelligence Index") or 0,
        ),
        reverse=True,
    )
    save_csv(rows, CSV_FIELDS, data_dir / "artificial_analysis_models.csv")

    # Summary
    with_evals = sum(
        1
        for m in models
        if m.get("evaluations")
        and any(v is not None for v in m["evaluations"].values())
    )
    creators = {}
    for m in models:
        c = (m.get("model_creator") or {}).get("name", "Unknown")
        creators[c] = creators.get(c, 0) + 1
    top = ", ".join(
        f"{n} ({c})" for n, c in sorted(creators.items(), key=lambda x: -x[1])[:5]
    )

    print_summary(
        "Artificial Analysis",
        len(models),
        **{
            "With evaluations": with_evals,
            "With pricing": sum(1 for m in models if m.get("pricing")),
            "With speed data": sum(
                1 for m in models if m.get("median_output_tokens_per_second")
            ),
            "Unique creators": len(creators),
            "Top creators": top,
        },
    )
    return models
