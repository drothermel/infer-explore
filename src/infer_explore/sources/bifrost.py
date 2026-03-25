"""Fetch model data from the Bifrost datasheet.

Pulls pricing, capability flags, and token limits for ~2000 chat models
across 72+ providers. Data is publicly available — no auth required.

Source: https://getbifrost.ai/datasheet
"""

import requests

from infer_explore.helpers import (
    get_data_dir,
    print_summary,
    save_csv,
    save_json,
)

DATA_JSON_URL = "https://getbifrost.ai/datasheet"


def fetch() -> dict:
    """Fetch the full datasheet JSON from Bifrost, filtered to chat models only."""
    print("Fetching models from Bifrost ...")
    resp = requests.get(DATA_JSON_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    total = len(data)

    # Filter to chat models only
    chat_models = {k: v for k, v in data.items() if v.get("mode") == "chat"}
    print(f"  Received {total} total entries, {len(chat_models)} chat models")
    return chat_models


def _flatten_model(key: str, entry: dict) -> dict:
    """Flatten one model entry into a CSV-friendly dict."""
    def _price_per_1m(val):
        if val is not None and val != 0:
            return val * 1_000_000
        return None

    modalities = entry.get("supported_modalities")
    modalities_str = ", ".join(modalities) if isinstance(modalities, list) else None

    return {
        "key": key,
        "base_model": entry.get("base_model", ""),
        "provider": entry.get("provider", ""),
        "input_price_per_1m": _price_per_1m(entry.get("input_cost_per_token")),
        "output_price_per_1m": _price_per_1m(entry.get("output_cost_per_token")),
        "cache_read_price_per_1m": _price_per_1m(entry.get("cache_read_input_token_cost")),
        "cache_write_price_per_1m": _price_per_1m(entry.get("cache_creation_input_token_cost")),
        "max_input_tokens": entry.get("max_input_tokens"),
        "max_output_tokens": entry.get("max_output_tokens"),
        "supports_function_calling": entry.get("supports_function_calling"),
        "supports_vision": entry.get("supports_vision"),
        "supports_reasoning": entry.get("supports_reasoning"),
        "supports_prompt_caching": entry.get("supports_prompt_caching"),
        "supports_web_search": entry.get("supports_web_search"),
        "supports_audio_input": entry.get("supports_audio_input"),
        "supports_audio_output": entry.get("supports_audio_output"),
        "supports_video_input": entry.get("supports_video_input"),
        "supports_pdf_input": entry.get("supports_pdf_input"),
        "supported_modalities": modalities_str,
        "deprecation_date": entry.get("deprecation_date"),
        "source": entry.get("source", ""),
    }


CSV_FIELDS = [
    "key",
    "base_model",
    "provider",
    "input_price_per_1m",
    "output_price_per_1m",
    "cache_read_price_per_1m",
    "cache_write_price_per_1m",
    "max_input_tokens",
    "max_output_tokens",
    "supports_function_calling",
    "supports_vision",
    "supports_reasoning",
    "supports_prompt_caching",
    "supports_web_search",
    "supports_audio_input",
    "supports_audio_output",
    "supports_video_input",
    "supports_pdf_input",
    "supported_modalities",
    "deprecation_date",
    "source",
]


def fetch_and_save() -> dict:
    """Fetch, save raw JSON + flattened CSV, print summary. Returns raw data."""
    chat_models = fetch()
    data_dir = get_data_dir()

    # Save raw JSON in standard envelope
    save_json(
        {"count": len(chat_models), "models": chat_models},
        data_dir / "bifrost_models.json",
        source_url=DATA_JSON_URL,
    )

    # Flatten into CSV rows
    rows = [_flatten_model(k, m) for k, m in sorted(chat_models.items())]
    save_csv(rows, CSV_FIELDS, data_dir / "bifrost_models.csv")

    # Summary stats
    providers = set(m.get("provider", "") for m in chat_models.values())
    base_models = set(m.get("base_model", "") for m in chat_models.values())
    with_pricing = sum(
        1 for m in chat_models.values() if m.get("input_cost_per_token") is not None
    )
    with_vision = sum(
        1 for m in chat_models.values() if m.get("supports_vision")
    )
    with_reasoning = sum(
        1 for m in chat_models.values() if m.get("supports_reasoning")
    )

    print_summary(
        "Bifrost",
        len(chat_models),
        **{
            "Unique providers": len(providers),
            "Unique base_models": len(base_models),
            "With pricing": with_pricing,
            "With vision": with_vision,
            "With reasoning": with_reasoning,
        },
    )
    return chat_models
