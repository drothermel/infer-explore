"""Fetch model data from the Vantage models site.

Pulls pricing across cloud vendors/regions, benchmark scores, and model metadata.
Data is publicly available at https://www.vantage.sh/models/data.json — no auth required.

Source: https://github.com/vantage-sh/models
"""

import requests

from infer_explore.helpers import (
    get_data_dir,
    print_summary,
    save_csv,
    save_json,
)

DATA_JSON_URL = "https://www.vantage.sh/models/data.json"


def fetch() -> dict:
    """Fetch the full data.json from vantage.sh/models."""
    print("Fetching models from Vantage ...")
    resp = requests.get(DATA_JSON_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Received {len(data.get('models', {}))} text models, "
          f"{len(data.get('imageModels', {}))} image models, "
          f"{len(data.get('vendors', {}))} vendors")
    return data


def _best_pricing(vendor_entry: dict) -> tuple[float | None, float | None]:
    """Extract the best (lowest) input and output price from a vendor's region pricing."""
    region_pricing = vendor_entry.get("regionPricing", {})
    input_prices = []
    output_prices = []
    for _region, prices in region_pricing.items():
        if isinstance(prices, list) and len(prices) >= 2:
            if prices[0] is not None:
                input_prices.append(prices[0])
            if prices[1] is not None:
                output_prices.append(prices[1])
    best_input = min(input_prices) if input_prices else None
    best_output = min(output_prices) if output_prices else None
    return best_input, best_output


def _flatten_model(model_key: str, m: dict, vendors_meta: dict) -> dict:
    """Flatten one model entry into a CSV-friendly dict."""
    # Find cheapest price across vendors
    best_input = None
    best_output = None
    vendor_names = []
    for v in m.get("vendors", []):
        vref = v.get("vendorRef", "")
        vendor_clean = vendors_meta.get(vref, {}).get("cleanName", vref)
        vendor_names.append(vendor_clean)
        inp, outp = _best_pricing(v)
        if inp is not None:
            # Convert per-token to per-1M-tokens
            inp_1m = inp * 1_000_000
            if best_input is None or inp_1m < best_input:
                best_input = inp_1m
        if outp is not None:
            outp_1m = outp * 1_000_000
            if best_output is None or outp_1m < best_output:
                best_output = outp_1m

    return {
        "key": model_key,
        "name": m.get("cleanName", ""),
        "company": m.get("company", ""),
        "country": m.get("companyCountryCode", ""),
        "release_date": m.get("releaseDate", ""),
        "training_cutoff": m.get("trainingCutoff", ""),
        "reasoning": m.get("reasoning"),
        "selfhostable": m.get("selfhostable"),
        "max_input_tokens": m.get("maxInputTokens"),
        "max_output_tokens": m.get("maxOutputTokens"),
        "best_input_price_per_1m": best_input,
        "best_output_price_per_1m": best_output,
        "num_vendors": len(m.get("vendors", [])),
        "vendors": ", ".join(vendor_names),
        "swe_bench_resolved_pct": m.get("sweBenchResolvedPercentage"),
        "swe_bench_cost_per_resolved": m.get("sweBenchCostPerResolved"),
        "skatebench_score": m.get("skatebenchScore"),
        "skatebench_cost_per_test": m.get("skatebenchCostPerTest"),
        "hle_pct": m.get("humanitysLastExamPercentage"),
    }


CSV_FIELDS = [
    "key",
    "name",
    "company",
    "country",
    "release_date",
    "training_cutoff",
    "reasoning",
    "selfhostable",
    "max_input_tokens",
    "max_output_tokens",
    "best_input_price_per_1m",
    "best_output_price_per_1m",
    "num_vendors",
    "vendors",
    "swe_bench_resolved_pct",
    "swe_bench_cost_per_resolved",
    "skatebench_score",
    "skatebench_cost_per_test",
    "hle_pct",
]


def fetch_and_save() -> dict:
    """Fetch, save raw JSON + flattened CSV, print summary. Returns raw data."""
    data = fetch()
    data_dir = get_data_dir()

    # Save raw JSON
    save_json(data, data_dir / "vantage_models.json", source_url=DATA_JSON_URL)

    # Flatten text models into CSV
    vendors_meta = data.get("vendors", {})
    models = data.get("models", {})
    rows = [_flatten_model(k, m, vendors_meta) for k, m in sorted(models.items())]
    save_csv(rows, CSV_FIELDS, data_dir / "vantage_models.csv")

    # Summary
    print_summary(
        "Vantage",
        len(models),
        **{
            "With release date": sum(1 for m in models.values() if m.get("releaseDate")),
            "With SWE-bench": sum(
                1 for m in models.values() if m.get("sweBenchResolvedPercentage")
            ),
            "With SkateBench": sum(
                1 for m in models.values() if m.get("skatebenchScore")
            ),
            "With HLE": sum(
                1 for m in models.values() if m.get("humanitysLastExamPercentage")
            ),
            "Vendors": ", ".join(
                v.get("cleanName", k) for k, v in vendors_meta.items()
            ),
        },
    )
    return data
