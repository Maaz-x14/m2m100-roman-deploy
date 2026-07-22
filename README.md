    

# Romanize API

Production FastAPI service for a fine-tuned M2M100 + LoRA model that transliterates Urdu script to Roman Urdu.

## Project structure

```
romanize_api/
├── app/                    # service code
│   ├── config.py           # all env-driven settings
│   ├── model.py            # model loading, patching, warm-up, transliterate()
│   ├── batcher.py          # dynamic request batching
│   ├── api.py               # /romanize and /health endpoints
│   └── main.py              # app factory, lifespan, uvicorn entry
├── fine_tuned_model/        # LoRA adapter weights (place here)
├── eval/
│   ├── layer1/               # rule-based evaluation of model outputs
│   └── layer2/               # LLM-judge evaluation (OpenAI API)
├── data/
│   ├── raw/                  # input dataset
│   ├── layer1/ layer2/       # evaluation outputs (gitignored)
│   └── wordlists/            # flagged/incorrect word lists
├── scripts/start.sh          # production startup script
├── diagrams/                 # architecture diagrams
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place the trained adapter directory at `fine_tuned_model/` (adapter_config.json, adapter_model.safetensors, tokenizer files). No config file is required — all settings have working defaults.

## Running

```bash
./scripts/start.sh
# or
uvicorn app.main:app --host 0.0.0.0 --port 2000 --workers 1
```

The server loads the model, merges LoRA weights, and runs warm-up inference before accepting traffic — watch the logs for "Romanize API is READY."

## API

**POST /romanize**

```bash
curl -X POST http://localhost:2000/romanize \
     -H "Content-Type: application/json" \
     -d '{"text": "آپ کیسے ہیں؟"}'
# → { "romanized_text": "Aap kaise hain?" }
```

Accepts a single string or a list of strings; response shape mirrors the input. List requests are processed as an explicit batch; single requests are grouped by the dynamic batcher.

| Status | Meaning                        |
| ------ | ------------------------------ |
| 400    | Blank or invalid input         |
| 503    | Model still loading            |
| 500    | Inference failure (check logs) |

**GET /health** — readiness probe, returns `{"status", "model_ready", "device"}`.

Interactive docs at `/docs` and `/redoc`.

## Environment variables

| Variable                 | Default                | Description                                            |
| ------------------------ | ---------------------- | ------------------------------------------------------ |
| `HOST`                 | `0.0.0.0`            | Bind address                                           |
| `PORT`                 | `2000`               | Listen port                                            |
| `MODEL_DIR`            | `./fine_tuned_model` | Path to LoRA adapter directory                         |
| `NUM_BEAMS`            | `4`                  | Beam search width                                      |
| `MAX_NEW_TOKENS`       | `256`                | Max Roman Urdu tokens per sentence                     |
| `INFERENCE_BATCH_SIZE` | `64`                 | Max sentences per generate() call / dynamic batch size |
| `MAX_WAIT_MS`          | `50`                 | Max wait before running a partial batch                |
| `WARMUP_SENTENCES`     | `3`                  | Dummy calls on startup                                 |
| `LOG_LEVEL`            | `INFO`               | Logging level                                          |

## Architecture notes

- **`workers=1`**: the merged model lives in VRAM; forking uvicorn workers would duplicate it. Scale horizontally, one container per GPU.
- **Dynamic batching** (`batcher.py`): concurrent single requests are collected into batches (size-or-time rule) before running `generate()`, since serving requests one at a time serializes GPU work. Tuned via sweep to `INFERENCE_BATCH_SIZE=64`, `MAX_WAIT_MS=50`.
- **`merge_and_unload()`**: LoRA adapter weights are baked into the base model at load time, removing per-request PEFT overhead.
- **No root endpoint**: `/` is intentionally unmounted; a 404 there signals a wrong URL rather than a generic response.

## Evaluation scripts

Separate from the deployed API — used to validate model output quality.

| Script                                | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `eval/layer1/eval_layer1.py`        | Rule-based pass over model outputs                           |
| `eval/layer1/inspect_layer1.py`     | Inspect/debug layer 1 results                                |
| `eval/layer2/eval_layer2_openai.py` | LLM-judge pass (OpenAI API) for phonetic/semantic mismatches |
| `eval/layer2/estimate_cost.py`      | Estimate OpenAI API cost before a full run                   |
| `eval/layer2/filter_issue_type.py`  | Filter layer 2 results by issue type                         |

Evaluation outputs (`layer1_results.csv`, `layer2_results.csv`, checkpoints, metrics) are run artifacts, not source-controlled — see `.gitignore`.
