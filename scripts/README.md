# Scripts

Data-fetching scripts for the infer-explore project.

## Setup

```bash
pip install -r scripts/requirements.txt
cp .env.example .env
# Edit .env and add your API keys
```

## fetch_artificial_analysis.py

Pulls model data from the [Artificial Analysis](https://artificialanalysis.ai/) free API:

- **Benchmark scores**: Intelligence Index, Coding Index, Math Index, MMLU-Pro, GPQA, HLE, LiveCodeBench, SciCode, MATH-500, AIME, and more
- **Pricing**: input/output/blended cost per 1M tokens
- **Speed**: median output tokens/sec, time to first token
- **Metadata**: model name, creator, release date, slug

```bash
python scripts/fetch_artificial_analysis.py
```

Outputs:
- `data/artificial_analysis_models.json` — raw API response (full detail)
- `data/artificial_analysis_models.csv` — flattened summary sorted by Intelligence Index

API docs: https://artificialanalysis.ai/api-reference#models-endpoint  
Rate limit: 1,000 requests/day (free tier)
