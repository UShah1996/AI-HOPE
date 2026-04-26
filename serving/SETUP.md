# AI-HOPE vLLM Serving — GPU Setup & Run Guide

## What this adds to AI-HOPE

The original AI-HOPE used Ollama (local CPU inference) for the multi-agent pipeline.
This `serving/` module replaces Ollama with vLLM for production-grade inference:

| Feature | Ollama | vLLM |
|---|---|---|
| Batching | None (sequential) | Continuous batching |
| Precision | FP32 | FP16 |
| KV cache | Per-request | Shared prefix caching |
| Concurrency | 1 request | N concurrent requests |
| Throughput | baseline | ~4× (measured) |

## Files

```
serving/
├── vllm_server.py      # vLLM AsyncEngine + OpenAI-compatible server
├── benchmark.py        # Throughput: vLLM vs HF Transformers baseline
├── kv_cache_demo.py    # Prefix KV cache sharing across planner/verifier
├── agents_vllm.py      # Drop-in replacement for Ollama-based agents
└── results/            # benchmark_results.json, kv_cache_demo.json
```

## Step 1 — Install vLLM (GPU required, CUDA 12+)

```bash
pip install vllm openai torch transformers
```

## Step 2 — Start the vLLM server

```bash
# Llama-2-7b in FP16 on single GPU
python serving/vllm_server.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --dtype float16 \
    --gpu-memory-utilization 0.90

# Verify it's up:
curl http://localhost:8000/v1/models
```

Note: requires HuggingFace token for Llama-2 access:
```bash
huggingface-cli login
```

## Step 3 — Run throughput benchmark

```bash
# In a second terminal while server is running:
python serving/benchmark.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --num_requests 100 \
    --concurrency 8
```

Expected output:
```
THROUGHPUT BENCHMARK RESULTS
  HF Transformers (FP16, sequential):     ~25 tok/s
  vLLM (continuous batching, FP16, c=8): ~100 tok/s
  Speedup: ~4×
  vLLM p99 latency: <200ms
```

## Step 4 — Run KV cache prefix demo

```bash
python serving/kv_cache_demo.py --model meta-llama/Llama-2-7b-chat-hf
```

This demonstrates that planner and verifier share KV blocks for the
shared system-prompt prefix — verifier TTFT is significantly lower
than a cold planner request because the prefix KV is already cached.

## Step 5 — Use vLLM backend in AI-HOPE

```python
# Replace this in src/app.py:
# from src.agents import run_pipeline

# With this:
from serving.agents_vllm import run_pipeline

result = run_pipeline(query, dataset_columns)
```

## Why the speedup is ~4×

HF Transformers processes requests sequentially: while one request is in the
generation phase, the GPU sits idle waiting for the next request. vLLM's
continuous batching scheduler fills those idle cycles by interleaving tokens
from multiple concurrent requests into a single GPU kernel call. Combined with
FP16 (2× memory bandwidth vs FP32), this produces ~4× throughput on typical
agent workloads.

Additionally, prefix caching means the N-token shared system prompt is only
computed once per GPU session — all subsequent planner/verifier turns skip
those tokens entirely, reducing effective prompt processing cost.
