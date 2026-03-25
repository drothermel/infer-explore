"""Merge all data sources into a unified model dataset.

Loads Artificial Analysis, HuggingFace, Vantage, and Bifrost JSON files,
extracts model identities, groups by model_id, and produces a single
unified JSON/CSV with deduplication.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from infer_explore.helpers import get_data_dir, save_csv, save_json
from infer_explore.sources.model_id import (
    ModelIdentity,
    extract_identity,
    _normalize_company,
)

# ---------------------------------------------------------------------------
# Company normalization (kept for backward compat; delegates to model_id)
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


def _norm_company(raw: str) -> str:
    """Normalize company/provider name to canonical form."""
    key = raw.strip().lower().replace(" ", "")
    return _COMPANY_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Model name normalization (kept for HF matching fallback)
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

        # Aggregate provider pricing with tiers and rate limits
        provider_pricing = []
        for key, entry in entries:
            inp = entry.get("input_cost_per_token")
            outp = entry.get("output_cost_per_token")
            if inp is not None or outp is not None:
                def _to_1m(val):
                    return val * 1_000_000 if val is not None else None

                prov = entry.get("provider", "")
                is_free = (inp is not None and inp == 0
                           and (outp is None or outp == 0))
                _SELF_HOSTED = {"ollama", "lemonade", "sagemaker"}
                tier = ("self-hosted" if prov in _SELF_HOSTED
                        else "free" if is_free else "paid")

                provider_pricing.append({
                    "provider": prov,
                    "key": key,
                    "input_price_per_1m": inp * 1_000_000 if inp else None,
                    "output_price_per_1m": outp * 1_000_000 if outp else None,
                    "cache_read_price_per_1m": (
                        entry.get("cache_read_input_token_cost", 0) or 0
                    ) * 1_000_000 or None,
                    "batch_input_price_per_1m": _to_1m(
                        entry.get("input_cost_per_token_batches")),
                    "batch_output_price_per_1m": _to_1m(
                        entry.get("output_cost_per_token_batches")),
                    "priority_input_price_per_1m": _to_1m(
                        entry.get("input_cost_per_token_priority")),
                    "priority_output_price_per_1m": _to_1m(
                        entry.get("output_cost_per_token_priority")),
                    "flex_input_price_per_1m": _to_1m(
                        entry.get("input_cost_per_token_flex")),
                    "flex_output_price_per_1m": _to_1m(
                        entry.get("output_cost_per_token_flex")),
                    "reasoning_output_price_per_1m": _to_1m(
                        entry.get("output_cost_per_reasoning_token")),
                    "rpm": entry.get("rpm"),
                    "tpm": entry.get("tpm"),
                    "is_free": is_free,
                    "tier": tier,
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
# Benchmark extraction helpers
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
# Source record wrappers — extract identity + data for each source record
# ---------------------------------------------------------------------------


def _process_aa_records(models: list[dict]) -> list[dict]:
    """Process AA models: extract identity and attach source data."""
    records = []
    for m in models:
        slug = m.get("slug", "")
        name = m.get("name", "")
        creator_raw = (m.get("model_creator") or {}).get("name", "")
        creator = _norm_company(creator_raw)

        identity = extract_identity(name or slug, creator=creator_raw, source="aa")

        records.append({
            "_identity": identity,
            "_source": "aa",
            "_name": name,
            "_raw": m,
        })
    return records


def _process_hf_records(models: list[dict]) -> list[dict]:
    """Process HF models: extract identity and attach source data."""
    records = []
    for m in models:
        model_name = m.get("model_name", "")
        provider = _norm_company(m.get("provider", ""))

        identity = extract_identity(model_name, creator=provider, source="hf")

        records.append({
            "_identity": identity,
            "_source": "hf",
            "_name": model_name,
            "_raw": m,
        })
    return records


def _process_vantage_records(models: dict[str, dict]) -> list[dict]:
    """Process Vantage models: extract identity and attach source data."""
    records = []
    for vkey, m in models.items():
        clean_name = m.get("cleanName", vkey)
        company = _norm_company(m.get("company", ""))

        identity = extract_identity(clean_name, creator=company, source="vantage")

        records.append({
            "_identity": identity,
            "_source": "vantage",
            "_name": clean_name,
            "_vantage_key": vkey,
            "_raw": m,
        })
    return records


def _process_bifrost_records(agg: dict[str, dict]) -> list[dict]:
    """Process aggregated Bifrost models: extract identity and attach source data.

    For Bifrost, the provider field refers to the *hosting* provider (e.g.
    gradient_ai, databricks), NOT the model creator (e.g. anthropic). We let
    the family detector infer the creator from the base_model name instead.
    """
    records = []
    for bm, entry in agg.items():
        # Use base_model as the primary name input (well-normalized)
        base_model = entry.get("_base_model", bm)

        # Don't pass the hosting provider as creator — let the family detector
        # infer the actual model creator from the name
        identity = extract_identity(base_model, creator="", source="bifrost")

        records.append({
            "_identity": identity,
            "_source": "bifrost",
            "_name": base_model,
            "_base_model": base_model,
            "_raw": entry,
        })
    return records


# ---------------------------------------------------------------------------
# Grouping and aggregation
# ---------------------------------------------------------------------------


def _group_by_model_id(all_records: list[dict]) -> dict[str, list[dict]]:
    """Group all source records by their extracted model_id."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in all_records:
        mid = rec["_identity"].model_id
        if mid:
            groups[mid].append(rec)
        else:
            # Fallback: use a slug-based ID
            fallback = _make_slug(rec["_name"], rec["_identity"].creator)
            groups[fallback].append(rec)
    return groups


def _build_unified_record(
    model_id: str,
    group: list[dict],
    vendors_meta: dict,
) -> dict:
    """Build a single unified record from a group of source records sharing a model_id.

    Aggregates pricing, benchmarks, capabilities, and configs from all sources.
    """
    identity = group[0]["_identity"]

    # Collect per-source data
    aa_records = [r for r in group if r["_source"] == "aa"]
    hf_records = [r for r in group if r["_source"] == "hf"]
    vn_records = [r for r in group if r["_source"] == "vantage"]
    bf_records = [r for r in group if r["_source"] == "bifrost"]

    # Pick best data source for each field
    aa = aa_records[0]["_raw"] if aa_records else None
    hf = hf_records[0]["_raw"] if hf_records else None
    vn = vn_records[0]["_raw"] if vn_records else None
    bf = bf_records[0]["_raw"] if bf_records else None

    # Sources present
    sources = []
    if aa_records:
        sources.append("aa")
    if hf_records:
        sources.append("hf")
    if vn_records:
        sources.append("vantage")
    if bf_records:
        sources.append("bifrost")

    # Best display name: AA > Vantage > identity.display_name > Bifrost
    name = ""
    if aa:
        name = aa.get("name", "")
    if not name and vn:
        name = vn.get("cleanName", "")
    if not name:
        name = identity.display_name
    if not name and bf:
        name = bf.get("_base_model", "")

    # Creator
    creator = identity.creator
    if not creator and aa:
        creator = _norm_company(
            (aa.get("model_creator") or {}).get("name", "")
        )
    if not creator and vn:
        creator = _norm_company(vn.get("company", ""))
    if not creator and hf:
        creator = _norm_company(hf.get("provider", ""))
    if not creator and bf:
        creator = _norm_company(bf.get("provider", ""))

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

    # Provider pricing: union from all Bifrost records in this group
    provider_pricing = []
    seen_pp_keys = set()
    for rec in bf_records:
        for pp in rec["_raw"].get("_provider_pricing", []):
            pp_key = pp.get("key", "")
            if pp_key not in seen_pp_keys:
                provider_pricing.append(pp)
                seen_pp_keys.add(pp_key)
    provider_pricing.sort(
        key=lambda p: (p.get("input_price_per_1m") or float("inf"))
    )

    # Compute price ranges from provider pricing
    paid_pp = [p for p in provider_pricing
               if p.get("input_price_per_1m") and p["input_price_per_1m"] > 0]
    free_pp = [p for p in provider_pricing if p.get("is_free")]
    min_input_price = min((p["input_price_per_1m"] for p in paid_pp), default=None)
    max_input_price = max((p["input_price_per_1m"] for p in paid_pp), default=None)
    min_output_price = min(
        (p["output_price_per_1m"] for p in paid_pp
         if p.get("output_price_per_1m") and p["output_price_per_1m"] > 0),
        default=None,
    )
    max_output_price = max(
        (p["output_price_per_1m"] for p in paid_pp
         if p.get("output_price_per_1m")),
        default=None,
    )
    num_free_providers = len(free_pp)
    num_paid_providers = len(paid_pp)
    has_batch = any(p.get("batch_input_price_per_1m") for p in provider_pricing)
    has_priority = any(p.get("priority_input_price_per_1m") for p in provider_pricing)
    has_flex = any(p.get("flex_input_price_per_1m") for p in provider_pricing)
    has_free_tier = num_free_providers > 0

    # Vendor pricing from Vantage
    vendor_pricing = _vantage_vendor_pricing(vn, vendors_meta) if vn else []

    # Context window / token limits — take max across all sources
    context_window = None
    max_output = None
    for rec in group:
        raw = rec["_raw"]
        if rec["_source"] == "bifrost":
            cw = raw.get("max_input_tokens")
            mo = raw.get("max_output_tokens")
        elif rec["_source"] == "vantage":
            cw = raw.get("maxInputTokens")
            mo = raw.get("maxOutputTokens")
        else:
            cw = None
            mo = None
        if cw and (context_window is None or cw > context_window):
            context_window = cw
        if mo and (max_output is None or mo > max_output):
            max_output = mo

    # Benchmarks: merge across all sources, preferring non-None
    all_benchmarks = []
    for rec in aa_records:
        all_benchmarks.append(_extract_aa_benchmarks(rec["_raw"]))
    for rec in hf_records:
        all_benchmarks.append(_extract_hf_benchmarks(rec["_raw"]))
    for rec in vn_records:
        all_benchmarks.append(_extract_vantage_benchmarks(rec["_raw"]))
    benchmarks = _merge_benchmarks(*all_benchmarks)

    # Capability flags — union across all Bifrost records (True wins)
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

    for rec in bf_records:
        raw = rec["_raw"]
        for field, current in [
            ("supports_vision", supports_vision),
            ("supports_function_calling", supports_function_calling),
            ("supports_web_search", supports_web_search),
            ("supports_pdf_input", supports_pdf_input),
            ("supports_audio_input", supports_audio_input),
            ("supports_audio_output", supports_audio_output),
            ("supports_video_input", supports_video_input),
            ("supports_prompt_caching", supports_prompt_caching),
            ("supports_computer_use", supports_computer_use),
            ("supports_reasoning", supports_reasoning),
        ]:
            val = raw.get(field)
            if val is True:
                locals()[field.replace("supports_", "supports_")] = True  # noqa

    # Re-read after loop since locals() trick doesn't work well
    for rec in bf_records:
        raw = rec["_raw"]
        if raw.get("supports_vision"):
            supports_vision = True
        if raw.get("supports_function_calling"):
            supports_function_calling = True
        if raw.get("supports_web_search"):
            supports_web_search = True
        if raw.get("supports_pdf_input"):
            supports_pdf_input = True
        if raw.get("supports_audio_input"):
            supports_audio_input = True
        if raw.get("supports_audio_output"):
            supports_audio_output = True
        if raw.get("supports_video_input"):
            supports_video_input = True
        if raw.get("supports_prompt_caching"):
            supports_prompt_caching = True
        if raw.get("supports_computer_use"):
            supports_computer_use = True
        if raw.get("supports_reasoning"):
            supports_reasoning = True
        mods = raw.get("supported_modalities")
        if isinstance(mods, list) and supported_modalities is None:
            supported_modalities = mods

    reasoning = supports_reasoning
    if reasoning is None and vn:
        reasoning = vn.get("reasoning")

    # Build configurations list from records that have config-level data
    configurations = []
    seen_configs = set()
    for rec in group:
        ident = rec["_identity"]
        cid = ident.config_id
        if cid in seen_configs:
            continue
        seen_configs.add(cid)

        config_entry = {
            "config_id": cid,
            "reasoning_mode": ident.reasoning_mode,
            "effort_level": ident.effort_level,
            "source_names": [rec["_name"]],
        }

        # Add config-specific benchmarks if from AA
        if rec["_source"] == "aa":
            config_entry["benchmarks"] = _extract_aa_benchmarks(rec["_raw"])
        elif rec["_source"] == "hf":
            config_entry["benchmarks"] = _extract_hf_benchmarks(rec["_raw"])

        configurations.append(config_entry)

    # Merge source_names for configs with same config_id
    # (already handled by seen_configs above, but add extra names)
    config_name_map = {c["config_id"]: c for c in configurations}
    for rec in group:
        cid = rec["_identity"].config_id
        if cid in config_name_map:
            if rec["_name"] not in config_name_map[cid]["source_names"]:
                config_name_map[cid]["source_names"].append(rec["_name"])

    # Collect all aliases
    aliases = []
    for rec in group:
        if rec["_name"] not in aliases:
            aliases.append(rec["_name"])

    # Num providers: sum across all bifrost records
    num_providers = sum(
        rec["_raw"].get("_num_providers", 0) for rec in bf_records
    )

    # For display name, strip reasoning/effort qualifiers from the best AA name
    display_name = name
    if "(" in display_name:
        # Try stripping the parenthesized qualifier to keep AA's casing
        cleaned = re.sub(r"\s*\([^)]*\)\s*", "", display_name).strip()
        if cleaned:
            display_name = cleaned
        else:
            display_name = identity.display_name or display_name

    record = {
        "id": model_id,
        "model_id": model_id,
        "name": display_name,
        "creator": creator,
        "family": identity.family,
        "variant": identity.variant,
        "version": identity.version,
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
        "min_input_price_per_1m": min_input_price,
        "max_input_price_per_1m": max_input_price,
        "min_output_price_per_1m": min_output_price,
        "max_output_price_per_1m": max_output_price,
        "num_free_providers": num_free_providers,
        "num_paid_providers": num_paid_providers,
        "has_batch_pricing": has_batch,
        "has_priority_pricing": has_priority,
        "has_flex_pricing": has_flex,
        "has_free_tier": has_free_tier,

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
        "num_providers": num_providers,
        "deprecation_date": bf.get("deprecation_date") if bf else None,

        # New fields
        "configurations": configurations,
        "num_configurations": len(configurations),
        "aliases": aliases,
    }
    return record


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------


def merge() -> list[dict]:
    """Load all sources, extract identities, group by model_id, and produce
    unified model list.
    """
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

    # Process each source: extract identities
    print("Extracting model identities...")
    all_records = []
    all_records.extend(_process_aa_records(aa_models))
    all_records.extend(_process_hf_records(hf_models))
    all_records.extend(_process_vantage_records(vantage_models))
    all_records.extend(_process_bifrost_records(bifrost_agg))

    print(f"  Total source records: {len(all_records)}")

    # Group by model_id
    print("Grouping by model_id...")
    groups = _group_by_model_id(all_records)
    print(f"  {len(all_records)} records → {len(groups)} unique model_ids")

    # Build unified records
    print("Building unified records...")
    unified = []
    for model_id, group in groups.items():
        record = _build_unified_record(model_id, group, vendors_meta)
        unified.append(record)

    # Sort by name
    unified.sort(key=lambda r: r["name"].lower())

    print(f"\nMerge complete: {len(unified)} unified models")
    src_counts = defaultdict(int)
    for r in unified:
        for s in r["sources"]:
            src_counts[s] += 1
    for src, count in sorted(src_counts.items()):
        print(f"  With {src}: {count}")
    multi = sum(1 for r in unified if len(r["sources"]) > 1)
    print(f"  Multi-source matches: {multi}")

    # Stats on dedup
    config_counts = [r["num_configurations"] for r in unified]
    multi_config = sum(1 for c in config_counts if c > 1)
    print(f"  Models with multiple configs: {multi_config}")

    return unified


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "id", "model_id", "name", "creator", "family", "variant", "version",
    "creator_country", "release_date",
    "model_type", "parameters_billions", "context_window", "max_output_tokens",
    "reasoning", "reasoning_tier", "selfhostable",
    "input_price_per_1m", "output_price_per_1m", "blended_price_per_1m",
    "min_input_price_per_1m", "max_input_price_per_1m",
    "min_output_price_per_1m", "max_output_price_per_1m",
    "cache_read_price_per_1m", "cache_write_price_per_1m",
    "num_free_providers", "num_paid_providers",
    "has_batch_pricing", "has_priority_pricing", "has_free_tier",
    "output_tokens_per_sec", "time_to_first_token_s",
    "supports_vision", "supports_function_calling", "supports_web_search",
    "supports_prompt_caching", "supports_pdf_input",
    "num_providers", "num_configurations", "sources",
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


def _sanitize_nans(obj):
    """Recursively replace NaN/Inf floats with None for JSON safety."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


def merge_and_save() -> list[dict]:
    """Merge all sources, save JSON + CSV, return unified list."""
    unified = merge()
    data_dir = get_data_dir()

    # Clean NaN values (from HuggingFace data) before saving
    unified = _sanitize_nans(unified)

    # Save JSON
    save_json(unified, data_dir / "merged_models.json", source_url="merged")

    # Save CSV
    rows = [_flatten_for_csv(r) for r in unified]
    save_csv(rows, CSV_FIELDS, data_dir / "merged_models.csv")

    return unified
