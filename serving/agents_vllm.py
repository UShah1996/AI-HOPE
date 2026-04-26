"""
src/agents_vllm.py — AI-HOPE Multi-Agent System (vLLM backend)

Drop-in replacement for the original Ollama-based agents.
Swap: ollama.chat() → vLLM OpenAI-compatible completions endpoint.

The three-agent pipeline is unchanged:
  Clarifier → Planner → Verifier → Statistical Engine

Only the inference backend changes — requests now go to the vLLM
server (serving/vllm_server.py) instead of local Ollama.
"""

import json
import os
import re
import time
from openai import OpenAI

# ── Client config ─────────────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL    = os.environ.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")

client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

# ── System prompts ─────────────────────────────────────────────────────────────
# Shared prefix goes first — vLLM prefix caching stores this block once
# and reuses it across clarifier/planner/verifier requests.
SHARED_PREFIX = (
    "You are an AI assistant for clinical and genomic data analysis. "
    "You operate as part of a multi-agent system. "
    "Always respond with valid JSON only. Never hallucinate column names."
)

SYSTEM_PROMPTS = {
    "clarifier": SHARED_PREFIX + (
        " Role: CLARIFIER. Check if the query is specific enough to execute. "
        "Output: {\"is_clear\": bool, \"ambiguities\": [], \"clarifying_questions\": []}"
    ),
    "planner": SHARED_PREFIX + (
        " Role: PLANNER. Convert the query into a structured JSON analysis plan. "
        "Output: {\"operation\": str, \"target_variable\": str, "
        "\"case_condition\": str, \"control_condition\": str}"
    ),
    "verifier": SHARED_PREFIX + (
        " Role: VERIFIER. Check planner JSON against actual dataset columns. "
        "Correct any hallucinated column names. "
        "Output: {\"verified\": bool, \"corrections\": {}, \"verified_plan\": {}}"
    ),
}

SAMPLING = {
    "clarifier": {"temperature": 0.2, "max_tokens": 128},
    "planner":   {"temperature": 0.1, "max_tokens": 256},
    "verifier":  {"temperature": 0.0, "max_tokens": 256},
}


# ── Base agent call ────────────────────────────────────────────────────────────
def _call_agent(role: str, user_message: str) -> dict:
    """Single agent call via vLLM OpenAI-compatible endpoint."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[role]},
            {"role": "user",   "content": user_message},
        ],
        **SAMPLING[role],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = response.choices[0].message.content.strip()

    # Parse JSON — strip markdown fences if present
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"raw": text, "parse_error": True}

    return {
        "role": role,
        "output": parsed,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


# ── Three-agent pipeline ───────────────────────────────────────────────────────
def run_pipeline(query: str, dataset_columns: list[str]) -> dict:
    """
    Full Clarifier → Planner → Verifier pipeline.

    All three agents share SHARED_PREFIX in their system prompts,
    so vLLM's prefix cache stores those KV blocks once and reuses
    them across all three turns.
    """
    results = {}

    # Agent 1: Clarifier
    clarifier_out = _call_agent("clarifier", query)
    results["clarifier"] = clarifier_out
    if not clarifier_out["output"].get("is_clear", True):
        return {
            "status": "needs_clarification",
            "questions": clarifier_out["output"].get("clarifying_questions", []),
            "pipeline": results,
        }

    # Agent 2: Planner
    planner_input = f"Query: {query}\nAvailable columns: {dataset_columns}"
    planner_out = _call_agent("planner", planner_input)
    results["planner"] = planner_out

    # Agent 3: Verifier
    verifier_input = (
        f"Planner output: {json.dumps(planner_out['output'])}\n"
        f"Actual dataset columns: {dataset_columns}\n"
        "Verify and correct any hallucinated column names."
    )
    verifier_out = _call_agent("verifier", verifier_input)
    results["verifier"] = verifier_out

    total_ms = sum(r["latency_ms"] for r in results.values())
    total_tokens = sum(
        r["prompt_tokens"] + r["completion_tokens"] for r in results.values()
    )

    return {
        "status": "success",
        "verified_plan": verifier_out["output"].get("verified_plan", {}),
        "pipeline": results,
        "total_latency_ms": round(total_ms, 2),
        "total_tokens": total_tokens,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    columns = [
        "patient_id", "age", "gender", "TUMOR_STAGE",
        "KRAS_mutation_status", "TP53_Mutation", "OS_STATUS", "OS_MONTHS"
    ]
    query = "Compare survival outcomes between KRAS mutant and wild-type patients"
    print(f"Query: {query}\n")
    result = run_pipeline(query, columns)
    print(json.dumps(result, indent=2))
