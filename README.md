# Offline Reconciliation Agent (FastAPI + FAISS + Local LLM)

This project provides an offline-first, local reconciliation agent that:

- Ingests CSV/JSON and normalizes the data.
- Generates sentence embeddings with a local `sentence-transformers` model.
- Indexes vectors using FAISS (cosine similarity).
- Uses an optional local LLM (llama.cpp via `llama-cpp-python`) for scoring and explainability.
- Implements reconciliation decisions: auto-merge, suggest-merge, no-merge.
- Exposes a FastAPI endpoint at `/reconcile` running on `http://localhost:8000` by default.

## Project Structure

- `data_ingest.py`: CSV/JSON ingest, normalization, and canonical text builder.
- `embeddings_index.py`: Embeddings generation and FAISS index build/query/save/load.
- `reconcile_agent.py`: Core reconciliation logic, thresholds, optional LLM scoring.
- `windsurf_skill.py`: FastAPI app exposing `/reconcile`.
- `agent_config.yaml`: Config for models, thresholds, API host/port.
- `requirements.txt`: Python dependencies.

## Quickstart

1) Create and activate a Python 3.10+ virtual environment.

```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2) Install dependencies.

```
pip install -r requirements.txt
```

3) Run the API.

```
python windsurf_skill.py
```

4) Test with example JSON.

```
curl -X POST http://127.0.0.1:8000/reconcile \
  -H "Content-Type: application/json" \
  -d @- << 'JSON'
{
  "records_a": [
    {"name": "Acme Corp", "address": "123 Main St", "city": "Springfield", "country": "US"},
    {"name": "Globex LLC", "address": "200 State Ave", "city": "Shelbyville", "country": "US"}
  ],
  "records_b": [
    {"name": "ACME Corporation", "address": "123 Main Street", "city": "Springfield", "country": "USA"},
    {"name": "Initech", "address": "300 Office Park", "city": "Springfield", "country": "US"}
  ],
  "text_fields": ["name", "address", "city", "country"]
}
JSON
```

Response includes a `summary`, `results`, and an `auto_merged_preview` of merged rows.

## Offline Model Preparation

### Sentence Embeddings (sentence-transformers)

- Default model: `sentence-transformers/all-MiniLM-L6-v2` (small, fast CPU model).
- To run fully offline, pre-download the model and point `embeddings.cache_dir` in `agent_config.yaml`.

```
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=".\.models\\embeddings")
_ = m.encode(["warmup"])  # downloads into cache
```

Then set:

```
embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  cache_dir: ./.models/embeddings
```

Copy the cached folder to an offline machine if needed.

### Local LLM (llama.cpp)

- Install `llama-cpp-python` from `requirements.txt`.
- Download a GGUF quantized model (e.g., Llama 3.1 8B Instruct Q4_K_M) from trusted sources like the model hub.
- Place the file under `./.models/llama/YourModel.gguf` and set in `agent_config.yaml`:

```
reconciliation:
  llm_backend: llama_cpp
  llama_cpp:
    model_path: ./.models/llama/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
    n_ctx: 4096
    n_threads: 6
    n_gpu_layers: 0
```

#### Quantization Notes

- Preferred for CPU-only: Q4_K_M or Q5_K_M.
- For smaller memory footprint: Q4_0 or Q3_K_M (reduced quality).
- You can quantize yourself using llama.cpp with `quantize` tool on an FP16 model to GGUF. Refer to llama.cpp docs.

### GPT4All (optional)

- If you prefer GPT4All, install `gpt4all` and adjust the `reconcile_agent.py` to initialize that backend (scaffold currently targets llama.cpp).

## Example Prompts

The agent uses this template when `llm_backend=llama_cpp`:

```
You are a data reconciliation assistant. Given two records, you output a match score between 0 and 1 and a short explanation.

Record A: {record_a}
Record B: {record_b}
Base similarity: {sim}
Respond as JSON with keys score (0-1 float) and reason (string).
```

For human-in-the-loop, you can present `results` with `decision == "suggest-merge"` to a reviewer.

## Resource Optimization

- Use `all-MiniLM-L6-v2` for fast CPU inference. Batch size ~64.
- Limit `top_k` to 1-3 during retrieval.
- Use FAISS inner product with normalized vectors to approximate cosine.
- Cache the FAISS index under `.state/faiss_index` for reuse.
- For llama.cpp, use Q4_K_M for decent quality on CPU, reduce `n_ctx` if memory constrained.

## Testing

- Unit test ingestion and text canonicalization with small fixtures.
- Run the API locally and POST small payloads first. Verify thresholds in `agent_config.yaml`:
  - `auto_merge_threshold`: 0.90
  - `suggest_merge_threshold`: 0.70
- Adjust thresholds per domain to tune precision/recall.

## Example Data (CSV/JSON)

You can also use your own CSV/JSON files by loading with `data_ingest.load_path(path)` and passing DataFrames into `ReconcileAgent.reconcile(df_a, df_b, text_fields=...)`.
