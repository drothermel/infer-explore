# Scripts

Thin wrappers around the `infer_explore` library. You can also use
`uv run <entry-point>` directly (see below).

## Setup

```bash
uv sync                  # install deps + library
cp .env.example .env     # add your API keys
```

## Entry points (preferred)

```bash
uv run fetch-aa        # Artificial Analysis only
uv run fetch-vantage   # Vantage only
uv run fetch-hf        # HuggingFace OpenEvals only
uv run fetch-all       # all three sources
```

## Scripts (alternative)

```bash
uv run python scripts/fetch_artificial_analysis.py
uv run python scripts/fetch_vantage.py
uv run python scripts/fetch_huggingface.py
uv run python scripts/fetch_all.py
```

## Output

All data is written to `data/`:

| File | Source | Auth |
|------|--------|------|
| `artificial_analysis_models.json` | [Artificial Analysis API](https://artificialanalysis.ai/api-reference) | API key (free) |
| `artificial_analysis_models.csv` | ↑ | ↑ |
| `vantage_models.json` | [vantage.sh/models](https://www.vantage.sh/models/data.json) | None |
| `vantage_models.csv` | ↑ | ↑ |
| `huggingface_openevals_models.json` | [OpenEvals/leaderboard-data](https://huggingface.co/datasets/OpenEvals/leaderboard-data) | None |
| `huggingface_openevals_models.csv` | ↑ | ↑ |
