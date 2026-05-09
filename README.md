# SRCoD

This repository contains the minimal code needed to collect response matrices and train SRCoD.

## Contents

- `srcod/`: dataset loaders, prompt construction, model clients, response collection, and SRCoD.
- `scripts/collect_responses.py`: collect response matrices with Ollama or a generic OpenAI-compatible API.
- `scripts/train_srcod.py`: train SRCoD from a response matrix and an external LSMI metrics CSV.
- `scripts/verify_package.py`: verify externally provided data, labels, images, and packaging constraints.

This repository intentionally does not include dataset files, LSMI prior files, existing response matrices, API keys, or a `response_matrix/` directory.

## Data

Dataset files are distributed separately. After downloading or mounting the data, place it under `data/` with this structure:

```text
data/
  MANIFEST.json
  MMMU/
  ScienceQA/
  SEED-Bench/
```

The labels used by SRCoD are human-annotated labels.

## Install

```bash
pip install -r requirements.txt
```

## Verify Package

Run this after placing the external data under `data/`.

```bash
python scripts/verify_package.py
```

## Collect Responses

Ollama example:

```bash
python scripts/collect_responses.py \
  --dataset ScienceQA \
  --backend ollama \
  --models qwen2.5vl:7b \
  --max-items 2
```

OpenAI-compatible API example:

```bash
set OPENAI_API_KEY=your_api_key
python scripts/collect_responses.py \
  --dataset ScienceQA \
  --backend openai \
  --models gpt-4o-mini \
  --max-items 2
```

For compatible services, pass `--base-url` explicitly. No third-party endpoint or key is hard-coded in this package.

The response collector does not set `max_tokens` for OpenAI-compatible calls and does not set `num_predict` for Ollama calls.

## Train SRCoD

SRCoD requires an external LSMI metrics CSV. The LSMI prior files are intentionally not packaged.

```bash
python scripts/train_srcod.py \
  --responses-path outputs/ScienceQA_responses/responses.parquet \
  --lsmi-metrics-path path/to/question_level_metrics.csv \
  --labels-path data/ScienceQA/labels.csv \
  --epochs 300 \
  --output-dir outputs/ScienceQA_srcod
```
