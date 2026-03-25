"""Merge all data sources into a unified model dataset.

Loads Artificial Analysis, HuggingFace, Vantage, and Bifrost JSON files,
normalizes names, fuzzy-matches models across sources, and produces a
single unified JSON/CSV.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from infer_explore.helpers import get_data_dir, save_csv, save_json

# ---------------------------------------------------------------------------
# Company normalization
# ---------------------------------------------------------------------------

_COMPANY_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "gemini": "google",
    "vertex_ai": "google",
    "vertex-ai": "google",
    "meta": "meta",
    "meta-llama": "meta",
    "mistral": "mistral",
    "mistralai": "mistral",
    "deepseek": "deepseek",
    "deepseek-ai": "deepseek",
    "alibaba": "alibaba",
    "qwen": "alibaba",
    "dashscope": "alibaba",
    "xai": "xai",
    "x.ai": "xai",
    "cohere": "cohere",
    "amazon": "amazon",
    "amazon_nova": "amazon",
    "bedrock": "amazon",
    "nvidia": "nvidia",
    "nvidia_nim": "nvidia",
    "zai": "zai",
    "z ai": "zai",
    "ibm": "ibm",
    "watsonx": "ibm",
    "microsoft": "microsoft",
    "ai21": "ai21",
    "ai21_labs": "ai21",
    "together_ai": "together",
    "together-ai": "together",
    "fireworks_ai": "fireworks",
    "fireworks-ai": "fireworks",
    "groq": "groq",
    "perplexity": "perplexity",
    "databricks": "databricks",
    "samsung": "samsung",
    "writer": "writer",
    "internlm": "internlm",
    "reka": "reka",
    "inflection": "inflection",
    "zhipu": "zhipu",
    "minimax": "minimax",
}


def _normalize_company(raw: str) -> str:
    """Normalize company/provider name to canonical form."""
    key = raw.strip().lower().replace(" ", "")
    return _COMPANY_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------

_STRIP_SUFFIXES_RE = re.compile(
    r"("
    r"-\d{8}"           # date suffixes like -20240620
    r"|\s*\((?:x?high|low|medium)\)"  # quality tiers
    r"|-instruct"
    r"|-chat"
    r"|:free"
    r"|:latest"
    r"|:beta"
    r"|:nitro"
    r"|:floor"
    r"|:extended"
    r"|:thinking"
    r")+$",
    re.IGNORECASE,
)

_VENDOR_PREFIX_RE = re.compile(
    r"^(?:anthropic\.|meta\.|us\.|eu\.|ap\.|cohere\.|mistral\.|amazon\.|writer\.|ai21\.)"
)

_VERSION_SUFFIX_RE = re.compile(r"-v\d+:\d+$")


def _normalize_model_name(raw: str) -> str:
    """Normalize a model name for matching."""
    name = raw.strip().lower()
    # Strip provider prefix (everything before first /)
    if "/" in name:
        name = name.split("/", 1)[-1]
    # Strip vendor-specific prefixes
    name = _VENDOR_PREFIX_RE.sub("", name)
    # Strip version suffixes like -v1:0
    name = _VERSION_SUFFIX_RE.sub("", name)
    # Strip quality/date suffixes
    name = _STRIP_SUFFIXES_RE.sub("", name)
    # Normalize separators
    name = name.replace("_", "-").replace(" ", "-")
    # Collapse multiple dashes
    name = re.sub(r"-+", "-", name).strip("-")
    return name


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    """Load a JSON file, return the parsed content."""
    with open(path) as f:
        return json.load(f)


def _load_aa(data_dir: Path) -> list[dict]:
    """Load Artificial Analysis models."""
    path = data_dir / "artificial_analysis_models.json"
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping AA")
        return []
    data = _load_json(path)
    return data.get("models", [])


def _load_hf(data_dir: Path) -> list[dict]:
    """Load HuggingFace OpenEvals models."""
    path = data_dir / "huggingface_openevals_models.json"
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping HF")
        return []
    data = _load_json(path)
    return data.get("models", [])


def _load_vantage(data_dir: Path) -> dict[str, dict]:
    """Load Vantage models (keyed by slug)."""
    path = data_dir / "vantage_models.json"
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping Vantage")
        return {}
    data = _load_json(path)
    return data.get("models", {})


def _load_bifrost(data_dir: Path) -> dict[str, dict]:
    """Load Bifrost models (keyed by model identifier)."""
    path = data_dir / "bifrost_models.json"
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping Bifrost")
        return {}
    data = _load_json(path)
    return data.get("models", {})


# ---------------------------------------------------------------------------
# Bifrost aggregation: group entries by base_model
# ---------------------------------------------------------------------------


def _aggregate_bifrost(raw: dict[str, dict]) -> dict[str, dict]:
    """Group Bifrost entries by base_model, aggregating provider pricing.

    Returns a dict keyed by base_model with aggregated info.
    """
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, entry in raw.items():
        bm = entry.get("base_model") or key
        groups[bm].append((key, entry))

    result = {}
    for base_model, entries in groups.items():
        # Pick the canonical entry (prefer the one whose key == base_model,
        # else first non-bedrock/non-azure entry)
        canonical = entries[0][1]
        for key, entry in entries:
            if key == base_model:
                canonical = entry
                break
        for key, entry in entries:
            prov = entry.get("provider", "")
            if prov not in ("bedrock", "azure", "vertex_ai"):
                canonical = entry
                break

        # Aggregate provider pricing
        provider_pricing = []
        for key, entry in entries:
            inp = entry.get("input_cost_per_token")
            outp = entry.get("output_cost_per_token")
            if inp is not None or outp is not None:
                provider_pricing.append({
                    "provider": entry.get("provider", ""),
                    "input_price_per_1m": inp * 1_000_000 if inp else None,
                    "output_price_per_1m": outp * 1_000_000 if outp else None,
                    "cache_read_price_per_1m": (
                        entry.get("cache_read_input_token_cost", 0) or 0
                    ) * 1_000_000 or None,
                    "source_url": entry.get("source"),
                })

        # Sort by input price
        provider_pricing.sort(
            key=lambda p: (p["input_price_per_1m"] or float("inf"))
        )

        result[base_model] = {
            **canonical,
            "_provider_pricing": provider_pricing,
            "_num_providers": len(entries),
            "_base_model": base_model,
        }
    return result


# ---------------------------------------------------------------------------
# Vantage helpers
# ---------------------------------------------------------------------------


def _vantage_vendor_pricing(model: dict, vendors_meta: dict) -> list[dict]:
    """Extract vendor pricing from Vantage model entry."""
    pricing = []
    for v in model.get("vendors", []):
        vref = v.get("vendorRef", "")
        vendor_name = vendors_meta.get(vref, {}).get("cleanName", vref)
        region_pricing = v.get("regionPricing", {})
        best_inp = None
        best_outp = None
        for _region, prices in region_pricing.items():
            if isinstance(prices, list) and len(prices) >= 2:
                if prices[0] is not None:
                    val = prices[0] * 1_000_000
                    if best_inp is None or val < best_inp:
                        best_inp = val
                if prices[1] is not None:
                    val = prices[1] * 1_000_000
                    if best_outp is None or val < best_outp:
                        best_outp = val
        if best_inp is not None or best_outp is not None:
            pricing.append({
                "vendor": vendor_name,
                "input_price_per_1m": best_inp,
                "output_price_per_1m": best_outp,
            })
    return pricing


# ---------------------------------------------------------------------------
# Building normalized index for matching
# ---------------------------------------------------------------------------


def _build_aa_index(models: list[dict]) -> dict[str, dict]:
    """Build a normalized-name -> AA model index."""
    idx: dict[str, dict] = {}
    for m in models:
        slug = m.get("slug", "")
        name = m.get("name", "")
        creator = _normalize_company(
            (m.get("model_creator") or {}).get("name", "")
        )
        norm = _normalize_model_name(slug or name)
        key = f"{creator}/{norm}"
        # Prefer models with evaluations
        if key not in idx or (
            m.get("evaluations") and not idx[key].get("evaluations")
        ):
            idx[key] = m
    return idx


def _build_hf_index(models: list[dict]) -> dict[str, dict]:
    """Build a normalized-name -> HF model index."""
    idx: dict[str, dict] = {}
    for m in models:
        model_name = m.get("model_name", "")
        provider = _normalize_company(m.get("provider", ""))
        # model_name is like "meta-llama/Llama-3.1-405B"
        norm = _normalize_model_name(model_name)
        key = f"{provider}/{norm}"
        idx[key] = m
    return idx


def _build_vantage_index(
    models: dict[str, dict],
) -> dict[str, tuple[str, dict]]:
    """Build a normalized-name -> (vantage_key, model) index."""
    idx: dict[str, tuple[str, dict]] = {}
    for vkey, m in models.items():
        company = _normalize_company(m.get("company", ""))
        norm = _normalize_model_name(vkey)
        key = f"{company}/{norm}"
        idx[key] = (vkey, m)
    return idx


def _build_bifrost_index(
    agg: dict[str, dict],
) -> dict[str, tuple[str, dict]]:
    """Build a normalized-name -> (base_model, aggregated_entry) index."""
    idx: dict[str, tuple[str, dict]] = {}
    for bm, entry in agg.items():
        provider = _normalize_company(entry.get("provider", ""))
        norm = _normalize_model_name(bm)
        # Index by multiple possible keys
        for company in [provider, norm.split("-")[0] if "-" in norm else provider]:
            comp = _normalize_company(company)
            key = f"{comp}/{norm}"
            if key not in idx:
                idx[key] = (bm, entry)
        # Also index by just the normalized name (no company prefix)
        idx[f"_any_/{norm}"] = (bm, entry)
    return idx


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _try_match_name(
    norm_key: str,
    norm_name: str,
    company: str,
    index: dict,
    any_prefix: bool = False,
) -> str | None:
    """Try to match a normalized key against an index.
    Returns the index key if found, else None.
    """
    # Exact match
    if norm_key in index:
        return norm_key

    # Try with _any_ prefix for cross-company matching
    if any_prefix:
        any_key = f"_any_/{norm_name}"
        if any_key in index:
            return any_key

    # Try substring matching: check if any index key contains our name or vice versa
    for idx_key in index:
        idx_name = idx_key.split("/", 1)[-1] if "/" in idx_key else idx_key
        idx_company = idx_key.split("/", 1)[0] if "/" in idx_key else ""

        # Company must match for substring matching
        if idx_company != company and idx_company != "_any_":
            continue

        if len(norm_name) >= 4 and len(idx_name) >= 4:
            if norm_name in idx_name or idx_name in norm_name:
                return idx_key

    return None


# ---------------------------------------------------------------------------
# Building unified model records
# ---------------------------------------------------------------------------


def _extract_aa_benchmarks(m: dict) -> dict:
    """Extract benchmark scores from an AA model record."""
    evals = m.get("evaluations") or {}
    return {
        "aa_intelligence_index": evals.get("artificial_analysis_intelligence_index"),
        "aa_coding_index": evals.get("artificial_analysis_coding_index"),
        "aa_math_index": evals.get("artificial_analysis_math_index"),
        "mmlu_pro": evals.get("mmlu_pro"),
        "gpqa": evals.get("gpqa"),
        "hle": evals.get("hle"),
        "livecodebench": evals.get("livecodebench"),
        "scicode": evals.get("scicode"),
        "math_500": evals.get("math_500"),
        "aime": evals.get("aime"),
        "aime_2025": evals.get("aime_25"),
        "ifbench": evals.get("ifbench"),
        "lcr": evals.get("lcr"),
        "terminalbench_hard": evals.get("terminalbench_hard"),
        "tau2": evals.get("tau2"),
    }


def _extract_hf_benchmarks(m: dict) -> dict:
    """Extract benchmark scores from an HF model record."""
    return {
        "aime_2026": m.get("aime2026_score"),
        "gsm8k": m.get("gsm8k_score"),
        "hmmt_2026": m.get("hmmt2026_score"),
        "olmocr": m.get("olmOcr_score"),
        "swe_pro": m.get("swePro_score"),
        "swe_bench_verified": m.get("sweVerified_score"),
        "evasion_bench": m.get("evasionBench_score"),
        "hle": m.get("hle_score"),
        "gpqa": m.get("gpqa_score"),
        "mmlu_pro": m.get("mmluPro_score"),
        "terminalbench_hard": m.get("terminalBench_score"),
    }


def _extract_vantage_benchmarks(m: dict) -> dict:
    """Extract benchmark scores from a Vantage model record."""
    return {
        "swe_bench_verified": m.get("sweBenchResolvedPercentage"),
        "skatebench": m.get("skatebenchScore"),
        "hle": m.get("humanitysLastExamPercentage"),
    }


def _merge_benchmarks(*benchmark_dicts: dict) -> dict:
    """Merge multiple benchmark dicts, preferring non-None values.
    Earlier dicts have higher priority.
    """
    all_keys = [
        "aa_intelligence_index", "aa_coding_index", "aa_math_index",
        "mmlu_pro", "gpqa", "hle", "livecodebench", "scicode",
        "math_500", "aime", "aime_2025", "aime_2026", "ifbench",
        "lcr", "terminalbench_hard", "tau2", "gsm8k", "hmmt_2026",
        "olmocr", "swe_pro", "swe_bench_verified", "evasion_bench",
        "skatebench",
    ]
    merged = {k: None for k in all_keys}
    for bm in benchmark_dicts:
        for k, v in bm.items():
            if v is not None and merged.get(k) is None:
                merged[k] = v
    return merged


def _make_slug(name: str, creator: str) -> str:
    """Generate a URL-friendly slug from name and creator."""
    raw = f"{creator}/{name}" if creator else name
    slug = raw.lower()
    slug = re.sub(r"[^a-z0-9/-]", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------


def merge() -> list[dict]:
    """Load all sources, match, and produce unified model list."""
    data_dir = get_data_dir()

    print("Loading data sources...")
    aa_models = _load_aa(data_dir)
    hf_models = _load_hf(data_dir)
    vantage_models = _load_vantage(data_dir)
    bifrost_raw = _load_bifrost(data_dir)

    # Load vantage vendor metadata for pricing extraction
    vantage_path = data_dir / "vantage_models.json"
    vendors_meta = {}
    if vantage_path.exists():
        vdata = _load_json(vantage_path)
        vendors_meta = vdata.get("vendors", {})

    print(f"  AA: {len(aa_models)}, HF: {len(hf_models)}, "
          f"Vantage: {len(vantage_models)}, Bifrost: {len(bifrost_raw)}")

    # Aggregate Bifrost by base_model
    print("Aggregating Bifrost entries by base_model...")
    bifrost_agg = _aggregate_bifrost(bifrost_raw)
    print(f"  {len(bifrost_raw)} entries → {len(bifrost_agg)} unique base models")

    # Build indexes
    print("Building match indexes...")
    aa_idx = _build_aa_index(aa_models)
    hf_idx = _build_hf_index(hf_models)
    vn_idx = _build_vantage_index(vantage_models)
    bf_idx = _build_bifrost_index(bifrost_agg)

    # Track which entries from each source have been matched
    matched_hf = set()
    matched_vn = set()
    matched_bf = set()

    unified = []

    # Start from AA (richest data), then add unmatched from other sources
    print("Matching models across sources...")

    for aa_key, aa in aa_idx.items():
        company = aa_key.split("/")[0]
        norm_name = aa_key.split("/", 1)[1]

        sources = ["aa"]
        hf_data = None
        vn_data = None
        bf_data = None

        # Try to match HF
        hf_match = _try_match_name(aa_key, norm_name, company, hf_idx, any_prefix=True)
        if hf_match:
            hf_data = hf_idx[hf_match]
            matched_hf.add(hf_match)
            sources.append("hf")

        # Try to match Vantage
        vn_match = _try_match_name(aa_key, norm_name, company, vn_idx, any_prefix=True)
        if vn_match:
            _, vn_data = vn_idx[vn_match]
            matched_vn.add(vn_match)
            sources.append("vantage")

        # Try to match Bifrost
        bf_match = _try_match_name(aa_key, norm_name, company, bf_idx, any_prefix=True)
        if bf_match:
            _, bf_data = bf_idx[bf_match]
            matched_bf.add(bf_match)
            sources.append("bifrost")

        record = _build_record(aa, hf_data, vn_data, bf_data, sources, vendors_meta)
        unified.append(record)

    # Add unmatched Vantage models
    for vn_key, (vantage_slug, vn_model) in vn_idx.items():
        if vn_key in matched_vn:
            continue
        company = vn_key.split("/")[0]
        norm_name = vn_key.split("/", 1)[1]
        sources = ["vantage"]

        hf_data = None
        bf_data = None

        hf_match = _try_match_name(vn_key, norm_name, company, hf_idx, any_prefix=True)
        if hf_match and hf_match not in matched_hf:
            hf_data = hf_idx[hf_match]
            matched_hf.add(hf_match)
            sources.append("hf")

        bf_match = _try_match_name(vn_key, norm_name, company, bf_idx, any_prefix=True)
        if bf_match and bf_match not in matched_bf:
            _, bf_data = bf_idx[bf_match]
            matched_bf.add(bf_match)
            sources.append("bifrost")

        record = _build_record(None, hf_data, vn_model, bf_data, sources, vendors_meta)
        unified.append(record)

    # Add unmatched Bifrost models
    for bf_key, (base_model, bf_entry) in bf_idx.items():
        if bf_key in matched_bf or bf_key.startswith("_any_/"):
            continue
        company = bf_key.split("/")[0]
        norm_name = bf_key.split("/", 1)[1]
        sources = ["bifrost"]

        hf_data = None
        hf_match = _try_match_name(bf_key, norm_name, company, hf_idx, any_prefix=True)
        if hf_match and hf_match not in matched_hf:
            hf_data = hf_idx[hf_match]
            matched_hf.add(hf_match)
            sources.append("hf")

        record = _build_record(None, hf_data, None, bf_entry, sources, vendors_meta)
        unified.append(record)

    # Add unmatched HF models
    for hf_key, hf_model in hf_idx.items():
        if hf_key in matched_hf:
            continue
        record = _build_record(None, hf_model, None, None, ["hf"], vendors_meta)
        unified.append(record)

    # Deduplicate by id
    seen_ids = {}
    deduped = []
    for record in unified:
        rid = record["id"]
        if rid in seen_ids:
            # Merge sources
            existing = seen_ids[rid]
            for s in record["sources"]:
                if s not in existing["sources"]:
                    existing["sources"].append(s)
            continue
        seen_ids[rid] = record
        deduped.append(record)

    # Sort by name
    deduped.sort(key=lambda r: r["name"].lower())

    print(f"\nMerge complete: {len(deduped)} unified models")
    src_counts = defaultdict(int)
    for r in deduped:
        for s in r["sources"]:
            src_counts[s] += 1
    for src, count in sorted(src_counts.items()):
        print(f"  With {src}: {count}")
    multi = sum(1 for r in deduped if len(r["sources"]) > 1)
    print(f"  Multi-source matches: {multi}")

    return deduped


def _build_record(
    aa: dict | None,
    hf: dict | None,
    vn: dict | None,
    bf: dict | None,
    sources: list[str],
    vendors_meta: dict,
) -> dict:
    """Build a unified model record from matched source data."""
    # Best name: AA > Vantage > HF > Bifrost
    name = ""
    if aa:
        name = aa.get("name", "")
    if not name and vn:
        name = vn.get("cleanName", "")
    if not name and hf:
        name = hf.get("model_name", "")
    if not name and bf:
        name = bf.get("_base_model", "")

    # Creator
    creator = ""
    if aa:
        creator = _normalize_company(
            (aa.get("model_creator") or {}).get("name", "")
        )
    if not creator and vn:
        creator = _normalize_company(vn.get("company", ""))
    if not creator and hf:
        creator = _normalize_company(hf.get("provider", ""))
    if not creator and bf:
        creator = _normalize_company(bf.get("provider", ""))

    # Pricing: prefer AA
    pricing = (aa.get("pricing") or {}) if aa else {}
    input_price = pricing.get("price_1m_input_tokens")
    output_price = pricing.get("price_1m_output_tokens")
    blended_price = pricing.get("price_1m_blended_3_to_1")

    # If no AA pricing, try Bifrost canonical
    if input_price is None and bf:
        inp_tok = bf.get("input_cost_per_token")
        if inp_tok is not None:
            input_price = inp_tok * 1_000_000
        outp_tok = bf.get("output_cost_per_token")
        if outp_tok is not None:
            output_price = outp_tok * 1_000_000

    cache_read = None
    cache_write = None
    if bf:
        cr = bf.get("cache_read_input_token_cost")
        if cr:
            cache_read = cr * 1_000_000
        cw = bf.get("cache_creation_input_token_cost")
        if cw:
            cache_write = cw * 1_000_000

    # Provider pricing from Bifrost
    provider_pricing = bf.get("_provider_pricing", []) if bf else []

    # Vendor pricing from Vantage
    vendor_pricing = _vantage_vendor_pricing(vn, vendors_meta) if vn else []

    # Context window / token limits
    context_window = None
    max_output = None
    if bf:
        context_window = bf.get("max_input_tokens")
        max_output = bf.get("max_output_tokens")
    if not context_window and vn:
        context_window = vn.get("maxInputTokens")
    if not max_output and vn:
        max_output = vn.get("maxOutputTokens")

    # Benchmarks
    aa_bench = _extract_aa_benchmarks(aa) if aa else {}
    hf_bench = _extract_hf_benchmarks(hf) if hf else {}
    vn_bench = _extract_vantage_benchmarks(vn) if vn else {}
    benchmarks = _merge_benchmarks(aa_bench, hf_bench, vn_bench)

    # Capability flags
    supports_vision = None
    supports_function_calling = None
    supports_web_search = None
    supports_pdf_input = None
    supports_audio_input = None
    supports_audio_output = None
    supports_video_input = None
    supports_prompt_caching = None
    supports_computer_use = None
    supports_reasoning = None
    supported_modalities = None

    if bf:
        supports_vision = bf.get("supports_vision")
        supports_function_calling = bf.get("supports_function_calling")
        supports_web_search = bf.get("supports_web_search")
        supports_pdf_input = bf.get("supports_pdf_input")
        supports_audio_input = bf.get("supports_audio_input")
        supports_audio_output = bf.get("supports_audio_output")
        supports_video_input = bf.get("supports_video_input")
        supports_prompt_caching = bf.get("supports_prompt_caching")
        supports_computer_use = bf.get("supports_computer_use")
        supports_reasoning = bf.get("supports_reasoning")
        mods = bf.get("supported_modalities")
        if isinstance(mods, list):
            supported_modalities = mods

    reasoning = supports_reasoning
    if reasoning is None and vn:
        reasoning = vn.get("reasoning")

    record = {
        "id": _make_slug(name, creator),
        "name": name,
        "creator": creator,
        "creator_country": vn.get("companyCountryCode") if vn else None,
        "release_date": (aa.get("release_date") if aa else None)
            or (vn.get("releaseDate") if vn else None),
        "training_cutoff": vn.get("trainingCutoff") if vn else None,

        "model_type": (hf.get("model_type") if hf else None)
            or ("open" if (vn and vn.get("selfhostable")) else None),
        "parameters_billions": hf.get("parameters_billions") if hf else None,
        "architecture": hf.get("architecture") if hf else None,
        "license": hf.get("license") if hf else None,
        "context_window": context_window,
        "max_output_tokens": max_output,

        "reasoning": reasoning,
        "reasoning_tier": vn.get("reasoningTier") if vn else None,
        "selfhostable": vn.get("selfhostable") if vn else None,
        "supports_function_calling": supports_function_calling,
        "supports_vision": supports_vision,
        "supports_web_search": supports_web_search,
        "supports_pdf_input": supports_pdf_input,
        "supports_audio_input": supports_audio_input,
        "supports_audio_output": supports_audio_output,
        "supports_video_input": supports_video_input,
        "supports_prompt_caching": supports_prompt_caching,
        "supports_computer_use": supports_computer_use,
        "supported_modalities": supported_modalities,

        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "blended_price_per_1m": blended_price,
        "cache_read_price_per_1m": cache_read,
        "cache_write_price_per_1m": cache_write,
        "provider_pricing": provider_pricing,
        "vendor_pricing": vendor_pricing,

        "output_tokens_per_sec": aa.get("median_output_tokens_per_second") if aa else None,
        "time_to_first_token_s": aa.get("median_time_to_first_token_seconds") if aa else None,
        "time_to_first_answer_s": aa.get("median_time_to_first_answer_token") if aa else None,

        "benchmarks": benchmarks,

        "swe_bench_cost_per_resolved": (
            (vn.get("sweBenchCostPerResolved") if vn else None)
            or (aa.get("swe_bench_cost_per_resolved") if aa else None)
        ),
        "skatebench_cost_per_test": (
            (vn.get("skatebenchCostPerTest") if vn else None)
        ),

        "hf_aggregate_score": hf.get("aggregate_score") if hf else None,
        "hf_coverage_count": hf.get("coverage_count") if hf else None,

        "sources": sources,
        "num_providers": bf.get("_num_providers", 0) if bf else 0,
        "deprecation_date": bf.get("deprecation_date") if bf else None,
    }
    return record


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "id", "name", "creator", "creator_country", "release_date",
    "model_type", "parameters_billions", "context_window", "max_output_tokens",
    "reasoning", "reasoning_tier", "selfhostable",
    "input_price_per_1m", "output_price_per_1m", "blended_price_per_1m",
    "cache_read_price_per_1m", "cache_write_price_per_1m",
    "output_tokens_per_sec", "time_to_first_token_s",
    "supports_vision", "supports_function_calling", "supports_web_search",
    "supports_prompt_caching", "supports_pdf_input",
    "num_providers", "sources",
    "hf_aggregate_score",
]


def _flatten_for_csv(record: dict) -> dict:
    """Flatten a unified record for CSV output."""
    row = {k: record.get(k) for k in CSV_FIELDS}
    # Join sources list
    if isinstance(row.get("sources"), list):
        row["sources"] = ", ".join(row["sources"])
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def merge_and_save() -> list[dict]:
    """Merge all sources, save JSON + CSV, return unified list."""
    unified = merge()
    data_dir = get_data_dir()

    # Save JSON
    save_json(unified, data_dir / "merged_models.json", source_url="merged")

    # Save CSV
    rows = [_flatten_for_csv(r) for r in unified]
    save_csv(rows, CSV_FIELDS, data_dir / "merged_models.csv")

    return unified
