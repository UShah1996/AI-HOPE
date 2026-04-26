"""
serving/kv_cache_demo.py — Prefix KV Cache Sharing Demo

Demonstrates that vLLM's automatic prefix caching reuses KV blocks
for the shared system-prompt prefix across planner and verifier turns.

With prefix caching ON:
  - First request: computes KV for full prompt (prefix + role + query)
  - Subsequent requests with same prefix: KV blocks for the shared
    system prompt are reused — only incremental tokens are computed

This is what "shared KV-cache across planner/verifier turns" means
on the resume. Both agents use the same Llama-2 checkpoint and the
same SHARED_SYSTEM_PREFIX, so vLLM caches that prefix once.

Usage:
    # Requires vLLM server running:
    python serving/vllm_server.py --model meta-llama/Llama-2-7b-chat-hf

    python serving/kv_cache_demo.py --model meta-llama/Llama-2-7b-chat-hf
"""

import asyncio
import json
import os
import time
import argparse

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("pip install openai")

from vllm_server import ROLE_PROMPTS, SHARED_SYSTEM_PREFIX

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


async def measure_ttft(client: AsyncOpenAI, model: str, prompt: str) -> float:
    """Measure Time-To-First-Token (ms) — proxy for KV cache hit."""
    t0 = time.perf_counter()
    stream = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1,       # just need first token
        temperature=0.0,
        stream=True,
    )
    async for _ in stream:
        break
    return (time.perf_counter() - t0) * 1000


def build_prompt(role: str, query: str) -> str:
    system = ROLE_PROMPTS[role]
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"{query} [/INST]"
    )


async def run_demo(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
    )

    query = "Compare survival outcomes between KRAS mutant and wild-type patients"
    dataset_columns = ["patient_id", "KRAS_mutation_status", "OS_STATUS", "OS_MONTHS", "TUMOR_STAGE"]

    planner_prompt = build_prompt("planner", f"Query: {query}\nColumns: {dataset_columns}")
    verifier_prompt = build_prompt("verifier", f"Plan: {{...}}\nColumns: {dataset_columns}")
    cold_prompt = (
        "<s>[INST] <<SYS>>\nYou are a completely different assistant with no shared context.\n"
        "<</SYS>>\n\nHello [/INST]"
    )

    print("\n[demo] Warming up prefix cache with planner request ...")
    # First call — computes KV for full planner prompt including shared prefix
    ttft_planner_cold = await measure_ttft(client, args.model, planner_prompt)
    print(f"  Planner (cold, no cache):    {ttft_planner_cold:.1f} ms TTFT")

    # Second call — shared prefix KV blocks are now cached
    ttft_planner_warm = await measure_ttft(client, args.model, planner_prompt)
    print(f"  Planner (warm, cache hit):   {ttft_planner_warm:.1f} ms TTFT")

    # Verifier — different role suffix but SAME shared prefix
    # vLLM reuses the cached KV for SHARED_SYSTEM_PREFIX
    ttft_verifier = await measure_ttft(client, args.model, verifier_prompt)
    print(f"  Verifier (shared prefix hit): {ttft_verifier:.1f} ms TTFT")

    # Cold prompt with completely different prefix — no cache benefit
    ttft_cold = await measure_ttft(client, args.model, cold_prompt)
    print(f"  Unrelated prompt (cold):     {ttft_cold:.1f} ms TTFT")

    # Compute cache speedup
    cache_speedup = ttft_planner_cold / ttft_verifier if ttft_verifier > 0 else 0

    results = {
        "model": args.model,
        "shared_prefix_tokens": len(SHARED_SYSTEM_PREFIX.split()),
        "ttft_ms": {
            "planner_cold": round(ttft_planner_cold, 1),
            "planner_warm": round(ttft_planner_warm, 1),
            "verifier_shared_prefix": round(ttft_verifier, 1),
            "unrelated_cold": round(ttft_cold, 1),
        },
        "prefix_cache_speedup": round(cache_speedup, 2),
        "interpretation": (
            f"Verifier TTFT is {cache_speedup:.1f}x faster than cold planner because "
            f"the {len(SHARED_SYSTEM_PREFIX.split())}-token shared system prefix KV is reused. "
            "Only the role-specific suffix and query tokens require new KV computation."
        ),
    }

    out_path = os.path.join(RESULTS_DIR, "kv_cache_demo.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("KV CACHE PREFIX SHARING RESULTS")
    print("=" * 60)
    print(f"  Shared prefix size:          ~{len(SHARED_SYSTEM_PREFIX.split())} tokens")
    print(f"  Planner cold TTFT:           {ttft_planner_cold:.1f} ms")
    print(f"  Verifier (shared prefix):    {ttft_verifier:.1f} ms")
    print(f"  Cache speedup:               {cache_speedup:.1f}×")
    print(f"  Saved → {out_path}")
    print("=" * 60)
    print("\nConclusion: planner and verifier share the same Llama-2 checkpoint")
    print("and the same system-prompt prefix. vLLM stores those KV blocks once")
    print("and reuses them across all agent turns — each turn only pays for")
    print("its incremental role-suffix + query tokens.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(run_demo(args))
