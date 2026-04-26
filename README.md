# AI-HOPE: Multi-Agent LLM Platform for Clinical & Genomic Data Analysis

A production multi-agent LLM system using a Planner-Verifier architecture for autonomous clinical data querying, with a vLLM serving layer for high-throughput inference.

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-orange)](https://github.com/vllm-project/vllm)
[![Llama3](https://img.shields.io/badge/Model-Llama--3-green)](https://ai.meta.com/llama/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

AI-HOPE is an LLM-driven multi-agent system that allows domain experts to run clinical and genomic data analyses — survival analysis, association studies, case-control comparisons — using natural language queries. The system uses a three-agent pipeline (Clarifier → Planner → Verifier) to validate queries, generate structured analysis plans, and correct hallucinated column names before execution.

**Implementation based on:** [AI-HOPE: an AI-driven conversational agent for enhanced clinical and genomic data integration in precision medicine research](https://doi.org/10.1093/bioinformatics/btaf359), *Bioinformatics* 2025, 41(7), btaf359.

---

## Serving Infrastructure (vLLM)

The original system used Ollama for local inference. The `serving/` module replaces this with **vLLM** for production-grade throughput:

| Feature | Ollama (original) | vLLM (serving/) |
|---|---|---|
| Batching | Sequential | Continuous batching |
| Precision | FP32 | FP16 |
| KV cache | Per-request | Shared prefix caching |
| Concurrency | 1 | N concurrent agent loops |

### Throughput benchmark

```bash
# Install vLLM (GPU required, CUDA 12+)
pip install vllm openai torch transformers

# Run in-process benchmark (vLLM AsyncEngine vs HF Transformers baseline)
python serving/benchmark.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --num_requests 100 \
    --concurrency 8
```

Results are written to `serving/results/benchmark_results.json`. See `serving/SETUP.md` for full reproduction instructions.

---

### Prefix KV cache sharing across agent turns

`agents_vllm.py` redesigns the prompt structure specifically to enable vLLM prefix caching: all three agents share a byte-identical `SHARED_SYSTEM_PREFIX` at the start of their system prompts, with role-specific instructions appended as a suffix. Since the prefix is identical across all three agent calls, vLLM's automatic prefix caching stores those KV blocks once per session — each turn only computes KV for its incremental role-suffix and query tokens.

```
SHARED_SYSTEM_PREFIX  ← cached once, reused by all three agents
    + CLARIFIER_SUFFIX  ← incremental KV only
    + PLANNER_SUFFIX    ← incremental KV only
    + VERIFIER_SUFFIX   ← incremental KV only
```

This is a deliberate prompt engineering contribution on top of the original AI-HOPE architecture, which used separate per-agent system prompts (incompatible with prefix caching). The redesign is documented in `serving/agents_vllm.py`.

To measure the actual cache speedup:
```bash
python serving/kv_cache_demo.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Multi-Agent Architecture

```
User Query
    │
    ▼
┌─────────────┐   ambiguous?   ┌──────────────────┐
│  Clarifier  │ ─────────────► │ Ask User         │
│ (Gatekeeper)│                └──────────────────┘
└─────────────┘
    │ clear
    ▼
┌─────────────┐
│   Planner   │ ── raw JSON ──────────────────────┐
│  (Logic Gen)│                                   │
└─────────────┘                                   ▼
                                       ┌─────────────────────┐
                                       │      Verifier       │
                                       │ (Hallucination Check│
                                       │  + Auto-Correction) │
                                       └─────────────────────┘
                                                  │ verified JSON
                                                  ▼
                                       ┌─────────────────────┐
                                       │  Statistical Engine │
                                       └─────────────────────┘
                                                  │
                                                  ▼
                                               Output
```

**Agent 1 — Clarifier:** Checks if the query is specific enough. Halts and asks clarifying questions if ambiguous — never guesses.

**Agent 2 — Planner:** Converts the natural language query into a structured JSON analysis plan:
`{"operation": ..., "target_variable": ..., "case_condition": ..., "control_condition": ...}`

**Agent 3 — Verifier:** Compares the Planner's JSON against actual dataset columns. If the Planner hallucinates a column name (e.g., `KRAS_Status` instead of `KRAS_mutation_status`), the Verifier auto-corrects it before execution.

---

## Repo Structure

```
AI-HOPE/
├── src/
│   ├── app.py              # Streamlit interface (Ollama backend)
│   └── agents.py           # Original Ollama-based agent calls
├── serving/                # vLLM serving layer (production backend)
│   ├── vllm_server.py      # AsyncLLMEngine, FP16, prefix caching config
│   ├── benchmark.py        # Throughput: vLLM vs HF Transformers baseline
│   ├── kv_cache_demo.py    # Prefix KV cache sharing measurement
│   ├── agents_vllm.py      # Drop-in replacement using vLLM + shared prefix
│   └── SETUP.md            # GPU setup and run instructions
├── tests/
├── generate_data.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option A — Original (Ollama, CPU-friendly)

```bash
git clone https://github.com/UShah1996/AI-HOPE.git
cd AI-HOPE
pip install -r requirements.txt

# Pull Llama3 via Ollama
ollama run llama3

# Run app
streamlit run src/app.py
```

### Option B — vLLM serving (GPU required, CUDA 12+)

```bash
pip install vllm openai torch transformers

# Run using vLLM AsyncEngine (in-process, no HTTP server needed)
# In src/app.py, swap:
#   from src.agents import run_pipeline
# for:
#   from serving.agents_vllm import run_pipeline

# Or launch vLLM's built-in OpenAI-compatible server:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16
```

See `serving/SETUP.md` for full GPU setup, HuggingFace token requirements, and benchmark reproduction steps.

---

## Supported Analysis Types

**Case-Control Studies:**
```
"Compare the frequency of TP53_Mutation in Stage IV vs Stage I patients"
```

**Survival Analysis:**
```
"Perform survival analysis grouping patients by KRAS_mutation_status"
```

**Global Association Scans:**
```
"Run a global association scan to find all variables correlated with OS_STATUS"
```

**Reliability Tests (Multi-Agent):**
```
"Is the data good?"
→ Clarifier triggers: asks for specific clarifying questions

"Compare survival for KRAS_Status"
→ Verifier auto-corrects KRAS_Status → KRAS_mutation_status
```

---

## Data Format

```
data/
└── your_dataset/
    ├── README.txt       # Dataset overview
    ├── index.txt        # Column names available for analysis
    └── data_table.tsv   # Tab-delimited data (rows = samples)
```

---

## Reference

> AI-HOPE: an AI-driven conversational agent for enhanced clinical and genomic data integration in precision medicine research.
> *Bioinformatics*, 2025, 41(7), btaf359. https://doi.org/10.1093/bioinformatics/btaf359
