"""Fetch model benchmark data from the HuggingFace OpenEvals leaderboard dataset.

Pulls the pre-aggregated multi-benchmark Parquet file from
OpenEvals/leaderboard-data, which combines scores across 11 official
HuggingFace benchmarks into a single table.

No authentication required. No rate limit.

Source: https://huggingface.co/datasets/OpenEvals/leaderboard-data
Docs:   https://huggingface.co/docs/hub/en/leaderboard-data-guide
"""

import pandas as pd

from infer_explore.helpers import (
    get_data_dir,
    print_summary,
    save_csv,
    save_json,
)

PARQUET_URL = (
    "hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet"
)

# Score columns we expect (suffix: _score)
SCORE_COLUMNS = [
    "aime2026_score",
    "evasionBench_score",
    "gpqa_score",
    "gsm8k_score",
    "hle_score",
    "hmmt2026_score",
    "mmluPro_score",
    "olmOcr_score",
    "swePro_score",
    "sweVerified_score",
    "terminalBench_score",
]

# Friendly labels for CSV header
SCORE_LABELS = {
    "aime2026_score": "AIME 2026",
    "evasionBench_score": "EvasionBench",
    "gpqa_score": "GPQA",
    "gsm8k_score": "GSM8K",
    "hle_score": "HLE",
    "hmmt2026_score": "HMMT 2026",
    "mmluPro_score": "MMLU-Pro",
    "olmOcr_score": "OlmOCR",
    "swePro_score": "SWE-Pro",
    "sweVerified_score": "SWE-bench Verified",
    "terminalBench_score": "TerminalBench",
}

META_COLUMNS = [
    "model_id",
    "model_name",
    "provider",
    "model_type",
    "parameters_billions",
    "license",
    "context_window",
    "modality",
    "architecture",
]

AGGREGATE_COLUMNS = [
    "aggregate_score",
    "coverage_count",
    "coverage_percent",
]


def fetch() -> pd.DataFrame:
    """Fetch the OpenEvals leaderboard Parquet file into a DataFrame."""
    print("Fetching models from HuggingFace OpenEvals ...")
    df = pd.read_parquet(PARQUET_URL)
    print(f"  Received {len(df)} models, {len(df.columns)} columns")
    return df


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, replacing NaN with None."""
    records = df.where(df.notna(), None).to_dict(orient="records")
    return records


def fetch_and_save() -> pd.DataFrame:
    """Fetch, save raw JSON + flattened CSV, print summary. Returns DataFrame."""
    df = fetch()
    data_dir = get_data_dir()

    # Discover actual score columns present in the data
    actual_score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
    # Also pick up any score columns we don't have labels for
    # Exclude aggregate_score since it's in AGGREGATE_COLUMNS
    extra_score_cols = [
        c
        for c in df.columns
        if c.endswith("_score") and c not in SCORE_COLUMNS and c != "aggregate_score"
    ]
    all_score_cols = actual_score_cols + extra_score_cols

    # Sort by aggregate_score descending
    if "aggregate_score" in df.columns:
        df = df.sort_values("aggregate_score", ascending=False, na_position="last")

    # Save raw JSON (full records)
    records = _df_to_records(df)
    save_json(
        records,
        data_dir / "huggingface_openevals_models.json",
        source_url="https://huggingface.co/datasets/OpenEvals/leaderboard-data",
    )

    # Save CSV with friendly headers
    csv_cols_present = (
        [c for c in META_COLUMNS if c in df.columns]
        + all_score_cols
        + [c for c in AGGREGATE_COLUMNS if c in df.columns]
    )
    csv_df = df[csv_cols_present].copy()

    # Rename score columns to friendly labels
    rename_map = {}
    for col in all_score_cols:
        if col in SCORE_LABELS:
            rename_map[col] = SCORE_LABELS[col]
    csv_df = csv_df.rename(columns=rename_map)

    # Write CSV
    csv_path = data_dir / "huggingface_openevals_models.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV  → {csv_path.name} ({len(csv_df)} rows)")

    # Summary
    score_coverage = {}
    for col in all_score_cols:
        label = SCORE_LABELS.get(col, col)
        n = int(df[col].notna().sum())
        score_coverage[f"With {label}"] = n

    providers = df["provider"].nunique() if "provider" in df.columns else 0

    print_summary(
        "HuggingFace OpenEvals",
        len(df),
        **{
            "Unique providers": providers,
            **score_coverage,
        },
    )
    return df
