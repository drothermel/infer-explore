# infer-explore

A Python library and toolkit for fetching, exploring, and visualizing LLM model data from multiple sources.

## Features

- **Multi-source data fetching** — pull model metadata and benchmarks from three providers:
  - [Artificial Analysis](https://artificialanalysis.ai/) (~450 models) — pricing, context windows, throughput metrics
  - [Vantage](https://www.vantage.sh/) (~117 models) — provider pricing and model details
  - [HuggingFace OpenEvals](https://huggingface.co/datasets/OpenEvals/leaderboard-data) (~93 models) — benchmark scores across multiple evaluations
- **Interactive cost explorer** — D3.js scatter plot for comparing OpenRouter model pricing with provider coloring, log-scale axes, and a click-to-select detail panel

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone https://github.com/drothermel/infer-explore.git
cd infer-explore
uv sync

# Set up environment (only needed for Artificial Analysis)
cp .env.example .env
# Edit .env to add your ARTIFICIAL_ANALYSIS_API_KEY
```

## Data Fetching

Each source has a dedicated CLI command and a standalone script. All commands write JSON and CSV to `data/`.

### CLI commands (via uv)

```bash
uv run fetch-aa        # Artificial Analysis (requires API key)
uv run fetch-vantage   # Vantage (no auth required)
uv run fetch-hf        # HuggingFace OpenEvals (no auth required)
uv run fetch-all       # All three sources
```

### Standalone scripts

```bash
uv run scripts/fetch_artificial_analysis.py
uv run scripts/fetch_vantage.py
uv run scripts/fetch_huggingface.py
uv run scripts/fetch_all.py
```

## Data Sources

| Source | Auth | Rate Limits | Output Files |
|--------|------|-------------|-------------|
| Artificial Analysis | API key (`x-api-key` header) | 1,000 req/day | `data/artificial_analysis_models.{json,csv}` |
| Vantage | None | None | `data/vantage_models.{json,csv}` |
| HuggingFace OpenEvals | None | None | `data/huggingface_openevals.{json,csv}` |

## Library Usage

The fetch logic is also available as a Python library:

```python
from infer_explore.sources.artificial_analysis import fetch_models as fetch_aa
from infer_explore.sources.vantage import fetch_models as fetch_vantage
from infer_explore.sources.huggingface import fetch_models as fetch_hf

# Fetch raw data
aa_models = fetch_aa()          # returns list of dicts
vantage_models = fetch_vantage()  # returns list of dicts
hf_models = fetch_hf()          # returns list of dicts
```

## Project Structure

```
infer-explore/
├── src/infer_explore/       # Library package
│   ├── __init__.py
│   ├── cli.py               # CLI entry points
│   ├── helpers.py            # Shared utilities (env, save, summaries)
│   └── sources/
│       ├── artificial_analysis.py
│       ├── vantage.py
│       └── huggingface.py
├── scripts/                  # Standalone wrapper scripts
├── data/                     # Fetched data (JSON + CSV)
├── openrouter-cost-explorer/ # Interactive D3.js visualization
├── pyproject.toml
└── .env.example
```

## OpenRouter Cost Explorer

An interactive scatter plot for exploring OpenRouter model pricing. Open `openrouter-cost-explorer/index.html` in a browser.

- Log-scale axes for input and output cost per million tokens
- Provider-colored dots with distinct colors for providers with 3+ models
- Adjustable cost-cap sliders
- Click any model for a slide-in detail panel with full metadata
