"""Extract structured model identity from raw name strings.

Given a raw model name (from any source), extracts creator, family, variant,
version, reasoning mode, effort level, and composes canonical model_id and
config_id strings for deduplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ModelIdentity dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelIdentity:
    """Structured identity extracted from a raw model name."""

    model_id: str = ""          # e.g. "anthropic/claude-sonnet-4.6"
    creator: str = ""           # e.g. "anthropic"
    family: str = ""            # e.g. "claude"
    variant: str = ""           # e.g. "sonnet"
    version: str = ""           # e.g. "4.6"

    reasoning_mode: str | None = None    # "reasoning", "non-reasoning", etc.
    effort_level: str | None = None      # "low", "medium", "high", etc.
    special_variant: str | None = None   # "pro", "search", "audio", etc.
    date_tag: str | None = None          # "(Jun '24)", "20240620", etc.

    config_id: str = ""         # model_id + config dimensions
    display_name: str = ""      # Human-readable name

    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REASONING_MODES = {
    "reasoning", "non-reasoning", "adaptive", "adaptive reasoning",
    "adaptive-reasoning", "thinking", "non reasoning",
}

EFFORT_LEVELS = {
    "low", "medium", "high", "xhigh", "max", "minimal",
    "max effort", "max-effort", "low effort", "low-effort",
    "high effort", "high-effort",
}

# These create DIFFERENT model_ids (not configs)
SPECIAL_VARIANTS = {
    "mini", "nano", "pro", "codex", "search", "audio", "realtime",
    "lite", "turbo", "fast",
}

# Date patterns
_DATE_PAREN_RE = re.compile(
    r"\((?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*'?\d{2,4}"
    r"|"
    r"\d{2,4}\s*[-/]\s*\d{2,4}"
    r")\)",
    re.IGNORECASE,
)
_DATE_SUFFIX_RE = re.compile(
    r"[-_]?\d{4}[-_]?\d{2}[-_]?\d{2}$"
)
_DATE_SHORT_RE = re.compile(
    r"[-_](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_]\d{2,4}$",
    re.IGNORECASE,
)
_DATE_MMDD_RE = re.compile(r"[-_]\d{2}[-_]\d{2,4}$")

# Provider prefixes to strip
_PROVIDER_PREFIXES = [
    "anthropic/", "openai/", "google/", "meta/", "meta-llama/",
    "mistral/", "mistralai/", "deepseek/", "deepseek-ai/",
    "cohere/", "alibaba/", "qwen/", "dashscope/",
    "nvidia/", "microsoft/", "ai21/", "xai/",
    "bedrock/", "azure/", "vertex_ai/", "vertex-ai/",
    "together_ai/", "together-ai/", "fireworks_ai/", "fireworks-ai/",
    "groq/", "perplexity/", "openrouter/", "replicate/",
    "snowflake/", "watsonx/", "sagemaker/", "ollama/",
    "anyscale/", "deepinfra/", "novita/", "publicai/",
    "gradient_ai/", "heroku/", "github_copilot/",
]

# Company normalization map (reused from merged.py)
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
    "zhipu": "zai",
    "ibm": "ibm",
    "watsonx": "ibm",
    "microsoft": "microsoft",
    "microsoft azure": "microsoft",
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
    "minimax": "minimax",
    "nous research": "nousresearch",
    "nousresearch": "nousresearch",
    "swiss ai initiative": "swissai",
    "servicenow": "servicenow",
    "snowflake": "snowflake",
    "deep cogito": "deepcogito",
    "lgairesearch": "lgairesearch",
    "lg ai research": "lgairesearch",
}

# Family detection patterns — order matters (longer/more specific first)
_FAMILY_PATTERNS = [
    ("claude", re.compile(r"\bclaude\b", re.I)),
    ("gpt", re.compile(r"\bgpt[-\s]?", re.I)),
    ("chatgpt", re.compile(r"\bchatgpt\b", re.I)),
    ("o1", re.compile(r"\bo[13][-\s]", re.I)),
    ("gemini", re.compile(r"\bgemini\b", re.I)),
    ("llama", re.compile(r"\bllama\b", re.I)),
    ("qwen", re.compile(r"\bqwen", re.I)),
    ("deepseek", re.compile(r"\bdeepseek\b", re.I)),
    ("mistral", re.compile(r"\bmistral\b", re.I)),
    ("codestral", re.compile(r"\bcodestral\b", re.I)),
    ("command", re.compile(r"\bcommand\b", re.I)),
    ("glm", re.compile(r"\bglm[-\s]?", re.I)),
    ("phi", re.compile(r"\bphi[-\s]?", re.I)),
    ("grok", re.compile(r"\bgrok\b", re.I)),
    ("nova", re.compile(r"\bnova\b", re.I)),
    ("jamba", re.compile(r"\bjamba\b", re.I)),
    ("dbrx", re.compile(r"\bdbrx\b", re.I)),
    ("gemma", re.compile(r"\bgemma\b", re.I)),
    ("falcon", re.compile(r"\bfalcon\b", re.I)),
    ("titan", re.compile(r"\btitan\b", re.I)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_company(raw: str) -> str:
    """Normalize company/provider name to canonical form."""
    key = raw.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    # Try exact match first
    if key in _COMPANY_MAP:
        return _COMPANY_MAP[key]
    # Try with original casing variations
    lower = raw.strip().lower()
    if lower in _COMPANY_MAP:
        return _COMPANY_MAP[lower]
    return lower or ""


def _strip_provider_prefix(name: str) -> str:
    """Strip cloud provider / routing prefixes from a model name."""
    lower = name.lower()
    # Strip everything before the last / if it looks like a provider path
    # e.g. "bedrock/us-east-1/anthropic.claude-3-5-sonnet-v1:0" -> "anthropic.claude-3-5-sonnet-v1:0"
    if "/" in name:
        parts = name.split("/")
        # Take the last meaningful part
        name = parts[-1]

    # Strip vendor-specific dot prefixes like "anthropic.", "meta.", etc.
    name = re.sub(
        r"^(?:anthropic|meta|us|eu|ap|apac|global|cohere|mistral|amazon|writer|ai21)\.",
        "",
        name,
        flags=re.I,
    )
    # Strip version suffixes like -v1:0, -v2:0
    name = re.sub(r"-v\d+:\d+$", "", name)
    # Strip date suffixes from API names (e.g. -20240620)
    name = re.sub(r"-\d{8}(?:-|$)", "", name).rstrip("-")
    # Strip :free, :latest, :beta, :nitro, :floor, :extended, :thinking suffixes
    name = re.sub(r":(free|latest|beta|nitro|floor|extended|thinking)$", "", name, flags=re.I)
    return name.strip()


def _extract_date_tag(name: str) -> tuple[str, str | None]:
    """Extract and remove date tags from a name. Returns (cleaned_name, date_tag)."""
    # Parenthesized dates: (Jun '24), (Oct '24), (Mar' 25)
    m = _DATE_PAREN_RE.search(name)
    if m:
        date_tag = m.group(0)
        cleaned = name[:m.start()] + name[m.end():]
        return cleaned.strip(), date_tag

    # YYYYMMDD suffix or YYYY-MM-DD in slugs
    m = _DATE_SUFFIX_RE.search(name)
    if m:
        return name[:m.start()].rstrip("-_ "), m.group(0).strip("-_")

    # Month-year suffix: -may-24, -dec-2024
    m = _DATE_SHORT_RE.search(name)
    if m:
        return name[:m.start()].rstrip("-_ "), m.group(0).strip("-_")

    # MM-YY or MMDD suffix like -0324, -0528
    m = _DATE_MMDD_RE.search(name)
    if m:
        tag = m.group(0).strip("-_")
        # Only treat as date if it's a 4-digit code that looks like MMYY or MMDD
        if len(tag) == 4:
            return name[:m.start()].rstrip("-_ "), tag

    return name, None


def _extract_qualifiers(text: str) -> tuple[str, str | None, str | None]:
    """Extract reasoning mode and effort level from parenthesized qualifiers.

    Returns (cleaned_text, reasoning_mode, effort_level).
    """
    reasoning_mode = None
    effort_level = None

    # Match parenthesized qualifiers like "(Non-reasoning, High Effort)"
    # but skip date-like content "(June '24)", "(Mar '25)", etc.
    paren_match = re.search(r"\(([^)]+)\)", text)
    if paren_match:
        quals = paren_match.group(1).lower().strip()
        # Check for combined qualifier
        parts = [p.strip() for p in quals.split(",")]
        found_qualifier = False
        for part in parts:
            # Normalize the part
            part_norm = part.strip()
            if part_norm in REASONING_MODES or part_norm.replace("-", " ") in REASONING_MODES:
                reasoning_mode = _normalize_reasoning_mode(part_norm)
                found_qualifier = True
            elif part_norm in EFFORT_LEVELS or part_norm.replace("-", " ") in EFFORT_LEVELS:
                effort_level = _normalize_effort_level(part_norm)
                found_qualifier = True
            elif "effort" in part_norm:
                effort_level = _normalize_effort_level(part_norm)
                found_qualifier = True
            elif "reasoning" in part_norm or part_norm == "thinking":
                reasoning_mode = _normalize_reasoning_mode(part_norm)
                found_qualifier = True

        # Only remove the parenthesized content if it contained a qualifier
        # (don't strip date tags like "(June '24)")
        if found_qualifier:
            text = text[:paren_match.start()] + text[paren_match.end():]
            text = text.strip()

    # Also check for inline reasoning/effort keywords without parens
    lower = text.lower()
    if reasoning_mode is None:
        for kw in ["non-reasoning", "non reasoning", "reasoning", "thinking",
                    "adaptive reasoning", "adaptive-reasoning"]:
            if kw in lower:
                reasoning_mode = _normalize_reasoning_mode(kw)
                # Remove from text
                text = re.sub(re.escape(kw), "", text, flags=re.I).strip()
                break

    if effort_level is None:
        for kw in ["max effort", "max-effort", "high effort", "high-effort",
                    "low effort", "low-effort"]:
            if kw in lower:
                effort_level = _normalize_effort_level(kw)
                text = re.sub(re.escape(kw), "", text, flags=re.I).strip()
                break

    # Clean up extra whitespace and trailing punctuation
    text = re.sub(r"\s+", " ", text).strip().strip("-_,. ")

    return text, reasoning_mode, effort_level


def _normalize_reasoning_mode(raw: str) -> str:
    """Normalize reasoning mode to canonical form."""
    raw = raw.lower().strip().replace("-", " ")
    if raw in ("non reasoning", "non-reasoning"):
        return "non-reasoning"
    if raw in ("adaptive reasoning", "adaptive-reasoning", "adaptive"):
        return "adaptive"
    if raw == "thinking":
        return "reasoning"
    if raw == "reasoning":
        return "reasoning"
    return raw


def _normalize_effort_level(raw: str) -> str:
    """Normalize effort level to canonical form."""
    raw = raw.lower().strip().replace("-", " ").replace("effort", "").strip()
    if raw == "max":
        return "max"
    if raw == "minimal":
        return "minimal"
    return raw  # low, medium, high, xhigh


def _normalize_version(v: str) -> str:
    """Normalize version string: replace hyphens between digits with dots."""
    # Convert "4-5" → "4.5", "3-7" → "3.7" but keep "3.1" as is
    v = re.sub(r"(\d)-(\d)", r"\1.\2", v)
    return v


def _build_model_id(creator: str, family: str, variant: str, version: str,
                     special_variant: str | None = None) -> str:
    """Compose model_id from parts."""
    parts = [family]
    if variant:
        parts.append(variant)
    if version:
        parts.append(version)
    if special_variant:
        parts.append(special_variant)
    name_part = "-".join(parts)
    if creator:
        return f"{creator}/{name_part}"
    return name_part


def _build_config_id(model_id: str, reasoning_mode: str | None,
                      effort_level: str | None) -> str:
    """Build config_id by appending config dimensions to model_id."""
    config_id = model_id
    if reasoning_mode:
        config_id += f":{reasoning_mode}"
    if effort_level:
        config_id += f":{effort_level}"
    return config_id


# Families whose name is an acronym (preserve uppercase)
_UPPERCASE_FAMILIES = {"gpt", "glm", "phi", "dbrx"}

# Families where version comes before variant in display name
# e.g. "Gemini 2.5 Flash" not "Gemini Flash 2.5"
_VERSION_FIRST_FAMILIES = {"gemini", "llama"}


def _build_display_name(family: str, variant: str, version: str,
                         special_variant: str | None = None) -> str:
    """Build a clean display name."""
    if family.lower() in _UPPERCASE_FAMILIES:
        family_display = family.upper()
    else:
        family_display = family.title()

    parts = [family_display]
    version_first = family.lower() in _VERSION_FIRST_FAMILIES

    if version and variant:
        if version_first:
            parts.append(version)
            parts.append(variant.title())
        else:
            parts.append(variant.title())
            parts.append(version)
    elif variant:
        parts.append(variant.title())
    elif version:
        parts.append(version)
    if special_variant:
        parts.append(special_variant.title())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Family-specific parsers
# ---------------------------------------------------------------------------


def _parse_claude(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Claude model names.

    Handles both word orders:
    - "Claude Sonnet 4.6" (variant-first)
    - "Claude 4.5 Sonnet" (version-first, used by AA)
    - "claude-sonnet-4-5" (slug form)
    - "claude-haiku-4-5" (Bifrost base_model)
    """
    clean = name
    # Extract qualifiers first
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    # Extract date tag
    clean, date_tag = _extract_date_tag(clean)

    # Normalize: strip "claude" prefix, normalize separators
    norm = re.sub(r"^claude[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Known Claude variants
    claude_variants = {"opus", "sonnet", "haiku", "instant"}

    # Try to extract variant and version from normalized name
    # Pattern 1: "sonnet-4-6", "opus-4.5", "haiku-4-5"
    m = re.match(
        r"(opus|sonnet|haiku|instant)[-\s]*(\d+(?:[.\-]\d+)?)?",
        norm, re.I,
    )
    if m:
        variant = m.group(1).lower()
        version = _normalize_version(m.group(2) or "")
    else:
        # Pattern 2: "4.5 Sonnet", "4-5-sonnet", "3.7 Sonnet"
        m = re.match(
            r"(\d+(?:[.\-]\d+)?)[-\s]*(opus|sonnet|haiku|instant)?",
            norm, re.I,
        )
        if m:
            version = _normalize_version(m.group(1) or "")
            variant = (m.group(2) or "").lower()

    # Claude Instant is a special case (no numeric version, or version 1.x)
    if variant == "instant":
        version = version or "1"

    # For very old Claude models: "Claude 2.0", "Claude 2.1"
    if not variant and version:
        # Claude 2.x and 3.x without variant
        pass

    creator = creator or "anthropic"
    model_id = _build_model_id(creator, "claude", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("Claude", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="claude",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_gpt(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse GPT model names.

    Handles: "GPT-5.4", "GPT-5.4 mini", "GPT-5.4 Pro", "GPT-4o",
    "gpt-4o-mini", "chatgpt-4o-latest", "gpt-4o-search", "gpt-4o-audio",
    "GPT-3.5 Turbo", "GPT-4 Turbo"
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Handle ChatGPT prefix
    is_chatgpt = bool(re.match(r"chatgpt", clean, re.I))

    # Normalize: strip "gpt-" or "chatgpt-" prefix
    norm = re.sub(r"^(?:chat)?gpt[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Detect effort levels from parenthesized qualifiers already stripped,
    # but also handle inline: "GPT-5 (high)", "GPT-5 (low)"
    lower_norm = norm.lower()

    # Handle "o" series: o1, o3, o4-mini
    o_match = re.match(r"^(o[134])[-\s]*(.*)", norm, re.I)
    if o_match:
        version = o_match.group(1).lower()
        rest = o_match.group(2).strip("-_ ")
        # Check for special variants like mini, pro
        if rest:
            rest_lower = rest.lower()
            for sv in sorted(SPECIAL_VARIANTS, key=len, reverse=True):
                if rest_lower.startswith(sv):
                    special_variant = sv
                    rest = rest[len(sv):].strip("-_ ")
                    break
    else:
        # Standard GPT: "5.4", "4o", "4-turbo", "3.5 Turbo"
        m = re.match(
            r"(\d+(?:[.\-]\d+)?)\s*(o)?\s*(.*)",
            norm, re.I,
        )
        if m:
            version = _normalize_version(m.group(1))
            if m.group(2):
                version += "o"
            rest = m.group(3).strip("-_ ")

            # Parse rest for special variants and effort levels
            if rest:
                rest_lower = rest.lower().strip()
                # Check for special variant first
                for sv in sorted(SPECIAL_VARIANTS, key=len, reverse=True):
                    if rest_lower.startswith(sv):
                        special_variant = sv
                        rest = rest_lower[len(sv):].strip("-_ ")
                        break

                # Remaining might be effort level
                if rest and not effort_level:
                    rest_clean = rest.lower().strip()
                    if rest_clean in EFFORT_LEVELS or rest_clean.replace("-", " ") in EFFORT_LEVELS:
                        effort_level = _normalize_effort_level(rest_clean)

    # ChatGPT is a special variant
    if is_chatgpt:
        special_variant = "chatgpt"

    # Handle "GPT-4o-search", "GPT-4o-audio", "GPT-4o-realtime"
    if not special_variant and version:
        for sv in ["search", "audio", "realtime"]:
            if sv in name.lower() and sv not in (version or ""):
                special_variant = sv
                break

    creator = creator or "openai"
    model_id = _build_model_id(creator, "gpt", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("GPT", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="gpt",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_gemini(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Gemini model names.

    Handles: "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 2.0 Flash-Lite",
    "Gemini 1.5 Flash-8B", "gemini-2-5-flash-preview"
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Strip "gemini" prefix
    norm = re.sub(r"^gemini[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Known Gemini variants (size tiers)
    gemini_variants = {"pro", "flash", "ultra", "nano"}
    # Gemini special sub-variants that create different model_ids
    gemini_specials = {"lite", "8b"}

    # Pattern: "2.5 Flash", "2-5-flash", "1.0 Pro"
    m = re.match(
        r"(\d+(?:[.\-]\d+)?)\s*[-\s]*(.*)",
        norm, re.I,
    )
    if m:
        version = _normalize_version(m.group(1))
        rest = m.group(2).strip("-_ ")

        # Parse variant from rest
        rest_lower = rest.lower()
        for v in sorted(gemini_variants, key=len, reverse=True):
            if rest_lower.startswith(v):
                variant = v
                rest = rest_lower[len(v):].strip("-_ ")
                break

        # Check for sub-variants like "Lite", "8B"
        if rest:
            rest_lower = rest.lower().strip()
            for sv in gemini_specials:
                if rest_lower.startswith(sv):
                    special_variant = sv
                    rest = rest_lower[len(sv):].strip("-_ ")
                    break

        # Check for "preview" — NOT a different model_id, strip it
        if rest and "preview" in rest.lower():
            rest = re.sub(r"\bpreview\b", "", rest, flags=re.I).strip("-_ ")
            # We don't set special_variant for preview

        # Check for "experimental" — also strip
        if rest and "experimental" in rest.lower():
            rest = re.sub(r"\bexperimental\b", "", rest, flags=re.I).strip("-_ ")

        # Remaining might have reasoning/effort
        if rest and not reasoning_mode:
            for rm in REASONING_MODES:
                if rm in rest.lower():
                    reasoning_mode = _normalize_reasoning_mode(rm)
                    break

    # Flash-Lite is written as one word sometimes
    if not special_variant and "lite" in norm.lower() and variant == "flash":
        special_variant = "lite"

    # Flash-8B detection
    if not special_variant and "8b" in norm.lower() and variant == "flash":
        special_variant = "8b"

    # Thinking experimental → reasoning config
    if "thinking" in name.lower() and not reasoning_mode:
        reasoning_mode = "reasoning"

    creator = creator or "google"
    model_id = _build_model_id(creator, "gemini", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("Gemini", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="gemini",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_llama(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Llama model names.

    Handles: "Llama 4 Maverick", "Llama 4 Scout", "Llama 3.3 70B",
    "Llama 3.1 Instruct 405B", "llama-3-1-70b-instruct"
    Parameter counts (8B, 70B, 405B) are DIFFERENT model_ids.
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Strip "llama" prefix
    norm = re.sub(r"^llama[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None
    param_count = ""

    # Known Llama named variants
    llama_variants = {"maverick", "scout"}

    # Parse version and the rest
    m = re.match(r"(\d+(?:[.\-]\d+)?)\s*(.*)", norm, re.I)
    if m:
        version = _normalize_version(m.group(1))
        rest = m.group(2).strip("-_ ")
    else:
        rest = norm

    # Strip "instruct", "chat" suffixes
    rest = re.sub(r"\b(?:instruct|chat)\b", "", rest, flags=re.I).strip("-_ ")

    # Extract parameter count (8B, 70B, 405B, etc.)
    param_match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", rest)
    if param_match:
        param_count = param_match.group(1).lower() + "b"
        rest = rest[:param_match.start()] + rest[param_match.end():]
        rest = rest.strip("-_ ")

    # Check for named variants
    rest_lower = rest.lower().strip()
    for v in llama_variants:
        if v in rest_lower:
            variant = v
            rest_lower = rest_lower.replace(v, "").strip("-_ ")
            break

    # Vision detection
    if "vision" in rest_lower:
        special_variant = "vision"

    # If no named variant but has param count, use that as variant
    if not variant and param_count:
        variant = param_count

    creator = creator or "meta"
    model_id = _build_model_id(creator, "llama", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("Llama", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="llama",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_qwen(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Qwen model names.

    Handles: "Qwen3-235B-A22B", "Qwen3-Coder-480B-A35B", "Qwen2.5 Max",
    "Qwen3 8B", "Qwen3.5 27B", "qwen3-32b-instruct"
    Parameter counts are different model_ids. "-thinking" = reasoning config.
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Strip "qwen" prefix
    norm = re.sub(r"^qwen", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None
    param_spec = ""

    # Known Qwen sub-families
    qwen_subfamilies = {"coder", "vl", "omni", "max", "turbo", "next"}

    # Parse version: "3", "2.5", "1.5", "3.5"
    # Qwen versions are small numbers (< 10). Anything larger is a param count.
    # We need to handle both "Qwen3 235B" and "Qwen3.5 27B" forms.
    m = re.match(r"(\d+(?:\.\d+)?)\s*(.*)", norm, re.I)
    if m:
        candidate = m.group(1)
        rest_after = m.group(2).strip("-_ ")
        # Qwen versions are always < 10 (1, 1.5, 2, 2.5, 3, 3.5, etc.)
        if float(candidate) < 10:
            version = _normalize_version(candidate)
            rest = rest_after
        else:
            # Large number — this is actually a param count, not a version
            rest = norm
    else:
        rest = norm

    # Check for sub-family
    rest_lower = rest.lower()
    for sf in sorted(qwen_subfamilies, key=len, reverse=True):
        if rest_lower.startswith(sf):
            special_variant = sf
            rest = rest_lower[len(sf):].strip("-_ ")
            rest_lower = rest.lower()
            break

    # Strip "instruct" suffix
    rest = re.sub(r"\b(?:instruct)\b", "", rest, flags=re.I).strip("-_ ")

    # Extract parameter spec: "235B-A22B", "480B-A35B", "8B", "32B"
    param_match = re.match(
        r"(\d+(?:\.\d+)?)\s*[bB](?:[-\s]*[aA](\d+(?:\.\d+)?)\s*[bB])?\s*(.*)",
        rest, re.I,
    )
    if param_match:
        total_params = param_match.group(1).lower() + "b"
        active_params = param_match.group(2)
        if active_params:
            param_spec = f"{total_params}-a{active_params.lower()}b"
        else:
            param_spec = total_params
        rest = param_match.group(3).strip("-_ ")

    # Use param_spec as variant
    variant = param_spec

    # Check for "thinking" suffix → reasoning mode
    if "thinking" in rest.lower() and not reasoning_mode:
        reasoning_mode = "reasoning"
        rest = re.sub(r"\bthinking\b", "", rest, flags=re.I).strip("-_ ")

    # Check for "preview" in remaining
    if "preview" in rest.lower():
        rest = re.sub(r"\bpreview\b", "", rest, flags=re.I).strip("-_ ")

    # For Qwen Max/Turbo (no param count), variant is empty
    # special_variant carries the sub-family

    creator = creator or "alibaba"
    model_id = _build_model_id(creator, "qwen", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)

    # Build display name
    display_parts = ["Qwen" + version]
    if special_variant:
        display_parts.append(special_variant.title())
    if variant:
        display_parts.append(variant.upper())
    display_name = " ".join(display_parts)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="qwen",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_deepseek(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse DeepSeek model names.

    Handles: "DeepSeek-V3", "DeepSeek-R1", "DeepSeek V3.1", "DeepSeek-V2.5",
    "DeepSeek R1 Distill Llama 70B", "DeepSeek Coder V2"
    R1 is a distinct model family from V-series.
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Strip "deepseek" prefix
    norm = re.sub(r"^deepseek[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Detect R1 vs V-series
    norm_lower = norm.lower()

    if norm_lower.startswith("r1") or norm_lower.startswith("r-1"):
        # R1 family
        variant = "r1"
        rest = re.sub(r"^r[-\s]*1\s*", "", norm, flags=re.I).strip("-_ ")

        # Check for distill variants
        distill_match = re.match(
            r"(?:distill\s*)(\w+)\s*(\d+(?:\.\d+)?)\s*[bB]?",
            rest, re.I,
        )
        if distill_match:
            base = distill_match.group(1).lower()
            params = distill_match.group(2).lower()
            special_variant = f"distill-{base}-{params}b"
            rest = rest[distill_match.end():].strip("-_ ")
        elif "qwen3" in rest.lower():
            # "DeepSeek R1 0528 Qwen3 8B"
            q_match = re.search(r"qwen\d*\s*(\d+)\s*b", rest, re.I)
            if q_match:
                special_variant = f"qwen3-{q_match.group(1).lower()}b"

    elif norm_lower.startswith("v") or norm_lower.startswith("v-"):
        # V-series
        m = re.match(r"v[-\s]*(\d+(?:[.\-]\d+)?)\s*(.*)", norm, flags=re.I)
        if m:
            version = "v" + _normalize_version(m.group(1))
            rest = m.group(2).strip("-_ ")
            # Check for sub-variants: "Terminus", "Speciale"
            if rest:
                rest_lower = rest.lower()
                for sv in ["terminus", "speciale", "exp"]:
                    if sv in rest_lower:
                        special_variant = sv
                        break
    elif norm_lower.startswith("coder"):
        # Coder variant
        variant = "coder"
        rest = re.sub(r"^coder[-\s]*", "", norm, flags=re.I).strip("-_ ")
        m = re.match(r"v[-\s]*(\d+(?:[.\-]\d+)?)\s*(.*)", rest, flags=re.I)
        if m:
            version = "v" + _normalize_version(m.group(1))
            rest = m.group(2).strip("-_ ")
            if "lite" in rest.lower():
                special_variant = "lite"
    elif norm_lower.startswith("llm"):
        variant = "llm"
        rest = re.sub(r"^llm[-\s]*", "", norm, flags=re.I).strip("-_ ")
        # Extract param count
        param_match = re.search(r"(\d+)\s*[bB]", rest)
        if param_match:
            special_variant = param_match.group(1).lower() + "b"

    creator = creator or "deepseek"
    model_id = _build_model_id(creator, "deepseek", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)

    display_parts = ["DeepSeek"]
    if variant:
        display_parts.append(variant.upper() if variant in ("r1", "llm") else variant.title())
    if version:
        display_parts.append(version.upper())
    if special_variant:
        display_parts.append(special_variant.title())
    display_name = " ".join(display_parts)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="deepseek",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_mistral(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Mistral model names.

    Handles: "Mistral Large 3", "Mistral Small 3.1", "Mistral 7B Instruct",
    "Mistral Medium 3", "Mistral Saba", "mistral-large-2407"
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Strip "mistral" prefix
    norm = re.sub(r"^mistral[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Known Mistral size tiers
    mistral_variants = {"large", "medium", "small", "saba", "nemo"}

    rest = norm
    rest_lower = rest.lower()

    # Check for named variant first
    for v in sorted(mistral_variants, key=len, reverse=True):
        if rest_lower.startswith(v):
            variant = v
            rest = rest_lower[len(v):].strip("-_ ")
            rest_lower = rest.lower()
            break

    # Extract version
    m = re.match(r"(\d+(?:[.\-]\d+)?)\s*(.*)", rest, re.I)
    if m:
        version = _normalize_version(m.group(1))
        rest = m.group(2).strip("-_ ")

    # Check for param count (7B, 24B, etc.)
    param_match = re.search(r"(\d+)\s*[bB]\b", rest)
    if param_match and not variant:
        variant = param_match.group(1).lower() + "b"

    # Strip "instruct" suffix
    rest = re.sub(r"\b(?:instruct)\b", "", rest, flags=re.I).strip("-_ ")

    creator = creator or "mistral"
    model_id = _build_model_id(creator, "mistral", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("Mistral", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="mistral",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_command(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Cohere Command model names."""
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    norm = re.sub(r"^command[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Known Command variants
    command_variants = {"r", "r+", "a", "light", "nightly"}

    rest_lower = norm.lower().strip()
    for v in sorted(command_variants, key=len, reverse=True):
        if rest_lower.startswith(v):
            variant = v
            rest_lower = rest_lower[len(v):].strip("-_ ")
            break

    creator = creator or "cohere"
    model_id = _build_model_id(creator, "command", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("Command", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="command",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_glm(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse GLM model names.

    Handles: "GLM-4.5", "GLM-4.5V", "GLM-4.6", "GLM-5", "GLM-5-Turbo",
    "GLM-4.7-Flash"
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    norm = re.sub(r"^glm[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Extract version
    m = re.match(r"(\d+(?:[.\-]\d+)?)\s*(.*)", norm, re.I)
    if m:
        version = _normalize_version(m.group(1))
        rest = m.group(2).strip("-_ ")
    else:
        rest = norm

    # Check for V suffix (vision) → separate model
    if rest.lower().startswith("v"):
        variant = "v"
        rest = rest[1:].strip("-_ ")

    # Check for sub-variants
    rest_lower = rest.lower()
    for sv in ["flash", "turbo", "air"]:
        if sv in rest_lower:
            special_variant = sv
            break

    creator = creator or "zai"
    model_id = _build_model_id(creator, "glm", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)

    display_parts = ["GLM", version]
    if variant:
        display_parts.append(variant.upper())
    if special_variant:
        display_parts.append(special_variant.title())
    display_name = " ".join(display_parts)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="glm",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_grok(name: str, creator: str, metadata: dict | None) -> ModelIdentity:
    """Parse Grok model names."""
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    norm = re.sub(r"^grok[-\s]*", "", clean, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Parse version and rest
    m = re.match(r"(\d+(?:[.\-]\d+)?)\s*(.*)", norm, re.I)
    if m:
        version = _normalize_version(m.group(1))
        rest = m.group(2).strip("-_ ")
    else:
        rest = norm

    # Check for special variants
    rest_lower = rest.lower()
    for sv in ["mini", "fast", "code"]:
        if sv in rest_lower:
            special_variant = sv
            rest_lower = rest_lower.replace(sv, "").strip("-_ ")
            break

    # "Beta" in rest
    if "beta" in rest_lower:
        rest_lower = rest_lower.replace("beta", "").strip("-_ ")

    # Check for reasoning in remaining text
    if "reasoning" in rest_lower and not reasoning_mode:
        reasoning_mode = "reasoning"

    creator = creator or "xai"
    model_id = _build_model_id(creator, "grok", variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)
    display_name = _build_display_name("Grok", variant, version, special_variant)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family="grok",
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


def _parse_generic(name: str, creator: str, family: str,
                    metadata: dict | None) -> ModelIdentity:
    """Generic parser for less common models.

    Does basic normalization, strips provider prefixes, detects reasoning/thinking.
    """
    clean = name
    clean, reasoning_mode, effort_level = _extract_qualifiers(clean)
    clean, date_tag = _extract_date_tag(clean)

    # Basic normalization
    norm = clean.strip()
    # Strip known family prefix
    norm = re.sub(r"^" + re.escape(family) + r"[-\s]*", "", norm, flags=re.I).strip("-_ ")

    variant = ""
    version = ""
    special_variant = None

    # Try to extract a version number
    m = re.match(r"(\d+(?:[.\-]\d+)?)\s*(.*)", norm, re.I)
    if m:
        version = _normalize_version(m.group(1))
        rest = m.group(2).strip("-_ ")
    else:
        rest = norm

    # Use remaining text as variant (cleaned)
    if rest:
        variant = re.sub(r"\b(?:instruct|chat|preview|experimental)\b", "", rest, flags=re.I)
        variant = re.sub(r"[-\s]+", "-", variant).strip("-_ ").lower()

    model_id = _build_model_id(creator, family, variant, version, special_variant)
    config_id = _build_config_id(model_id, reasoning_mode, effort_level)

    display_parts = [family.title()]
    if version:
        display_parts.append(version)
    if variant:
        display_parts.append(variant.title())
    display_name = " ".join(display_parts)

    return ModelIdentity(
        model_id=model_id,
        creator=creator,
        family=family,
        variant=variant,
        version=version,
        reasoning_mode=reasoning_mode,
        effort_level=effort_level,
        special_variant=special_variant,
        date_tag=date_tag,
        config_id=config_id,
        display_name=display_name,
    )


# ---------------------------------------------------------------------------
# Family parser dispatch
# ---------------------------------------------------------------------------

_FAMILY_PARSERS = {
    "claude": _parse_claude,
    "gpt": _parse_gpt,
    "chatgpt": _parse_gpt,
    "o1": _parse_gpt,  # O-series → GPT parser
    "gemini": _parse_gemini,
    "llama": _parse_llama,
    "qwen": _parse_qwen,
    "deepseek": _parse_deepseek,
    "mistral": _parse_mistral,
    "codestral": lambda n, c, m: _parse_generic(n, c or "mistral", "codestral", m),
    "command": _parse_command,
    "glm": _parse_glm,
    "grok": _parse_grok,
}


def _detect_family(name: str) -> str | None:
    """Detect model family from name string."""
    lower = name.lower()
    # Special case: ChatGPT → gpt
    if lower.startswith("chatgpt"):
        return "chatgpt"
    # Special case: O-series (o1, o3, o4-mini) — but NOT "ollama", "open", etc.
    if re.match(r"^o[134][-\s]", lower) or re.match(r"^o[134]$", lower):
        return "o1"
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(lower):
            return family
    return None


def _infer_creator(family: str, creator: str) -> str:
    """Infer creator from family if not provided."""
    family_to_creator = {
        "claude": "anthropic",
        "gpt": "openai",
        "chatgpt": "openai",
        "o1": "openai",
        "gemini": "google",
        "llama": "meta",
        "qwen": "alibaba",
        "deepseek": "deepseek",
        "mistral": "mistral",
        "codestral": "mistral",
        "command": "cohere",
        "glm": "zai",
        "grok": "xai",
        "phi": "microsoft",
        "nova": "amazon",
        "jamba": "ai21",
        "dbrx": "databricks",
        "gemma": "google",
        "falcon": "tii",
        "titan": "amazon",
    }
    if creator:
        return _normalize_company(creator)
    return family_to_creator.get(family, "")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_identity(
    name: str,
    creator: str = "",
    source: str = "",
    metadata: dict | None = None,
) -> ModelIdentity:
    """Extract structured model identity from a raw name string.

    Args:
        name: Raw model name (e.g. "Claude Sonnet 4.6 (Non-reasoning, High Effort)")
        creator: Creator/company name if known
        source: Source identifier (aa, bifrost, vantage, hf)
        metadata: Additional metadata dict

    Returns:
        ModelIdentity with extracted fields
    """
    if not name:
        return ModelIdentity()

    # Step 1: Strip provider prefixes
    clean = _strip_provider_prefix(name)

    # Step 2: Detect family
    family = _detect_family(clean)
    if not family:
        # Try with original name too
        family = _detect_family(name)

    # Step 3: Infer creator
    resolved_creator = _infer_creator(family or "", creator)

    # Step 4: Dispatch to family parser
    if family and family in _FAMILY_PARSERS:
        parser = _FAMILY_PARSERS[family]
        identity = parser(clean, resolved_creator, metadata)
    else:
        # Generic fallback
        identity = _parse_generic(
            clean, resolved_creator, family or _guess_family_from_name(clean), metadata
        )

    # Ensure aliases includes the original name
    if name not in identity.aliases:
        identity.aliases.append(name)

    return identity


def _guess_family_from_name(name: str) -> str:
    """Last resort: use the first word as family name."""
    first = re.split(r"[-\s_/]+", name.strip())[0].lower()
    return first or "unknown"
