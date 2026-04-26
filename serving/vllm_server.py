"""
serving/vllm_server.py — AI-HOPE vLLM Serving Layer

Replaces the Ollama backend with vLLM for production inference.
Key features:
  - Continuous batching (vLLM scheduler fills gaps between requests)
  - FP16 quantization (dtype=float16, ~2x memory vs FP32)
  - Automatic prefix caching — planner and verifier share the same
    system-prompt prefix, so vLLM reuses those KV blocks across turns
  - AsyncEngine for concurrent agent loops without blocking

Usage (start server):
    python serving/vllm_server.py --model meta-llama/Llama-2-7b-chat-hf \
        --port 8000 --dtype float16 --gpu-memory-utilization 0.9

Usage (as library):
    from serving.vllm_server import AIHOPEInferenceEngine
    engine = AIHOPEInferenceEngine(model="meta-llama/Llama-2-7b-chat-hf")
    response = await engine.generate(prompt="...", role="planner")
"""

import argparse
import asyncio
import time
import os
from dataclasses import dataclass, field
from typing import AsyncIterator

# vLLM imports
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid


# ── System prompts (shared prefix → KV cache reuse) ───────────────────────────
# Both planner and verifier start with the same prefix block.
# vLLM's automatic prefix caching stores this once and reuses it across
# all concurrent agent requests — no redundant KV computation.

SHARED_SYSTEM_PREFIX = """You are an AI assistant for clinical and genomic data analysis.
You operate as part of a multi-agent system analyzing tabular datasets.
Available operations: prevalence_test, survival_analysis, association_scan, odds_ratio.
Always respond with valid JSON. Never hallucinate column names."""

PLANNER_SUFFIX = """
Role: PLANNER. Convert the user's natural language query into a structured JSON analysis plan.
Output format: {"operation": str, "target_variable": str, "case_condition": str, "control_condition": str}
"""

VERIFIER_SUFFIX = """
Role: VERIFIER. Check the planner's JSON against the actual dataset columns.
If any column name is hallucinated, correct it to the closest real column name.
Output format: {"verified": bool, "corrections": {}, "verified_plan": {}}
"""

CLARIFIER_SUFFIX = """
Role: CLARIFIER. Assess if the query is specific enough to execute.
If ambiguous, output the clarifying questions needed.
Output format: {"is_clear": bool, "ambiguities": [], "clarifying_questions": []}
"""

ROLE_PROMPTS = {
    "planner": SHARED_SYSTEM_PREFIX + PLANNER_SUFFIX,
    "verifier": SHARED_SYSTEM_PREFIX + VERIFIER_SUFFIX,
    "clarifier": SHARED_SYSTEM_PREFIX + CLARIFIER_SUFFIX,
}


# ── Sampling configs per role ─────────────────────────────────────────────────
ROLE_SAMPLING = {
    "planner":   SamplingParams(temperature=0.1, max_tokens=256, stop=["}\n\n"]),
    "verifier":  SamplingParams(temperature=0.0, max_tokens=256, stop=["}\n\n"]),
    "clarifier": SamplingParams(temperature=0.2, max_tokens=128, stop=["}\n\n"]),
}


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    role: str
    prompt_tokens: int
    output_tokens: int
    text: str
    latency_ms: float
    request_id: str


# ── Engine ────────────────────────────────────────────────────────────────────
class AIHOPEInferenceEngine:
    """
    Wraps vLLM AsyncLLMEngine for multi-agent AI-HOPE workloads.

    Key design choices:
    - dtype=float16: halves KV cache memory vs bfloat16/float32
    - enable_prefix_caching=True: planner/verifier share system-prompt
      KV blocks — each agent turn only pays for its incremental tokens
    - gpu_memory_utilization=0.90: leaves headroom for CUDA kernels
    - max_num_seqs controls how many agent loops run concurrently
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 64,
        tensor_parallel_size: int = 1,
    ):
        print(f"[engine] Initialising vLLM engine: {model}")
        print(f"[engine] dtype={dtype}  gpu_mem={gpu_memory_utilization}  max_seqs={max_num_seqs}")

        engine_args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,   # KV cache reuse across shared prefixes
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model = model
        print("[engine] Ready")

    def _build_prompt(self, role: str, user_message: str) -> str:
        """Format prompt with Llama-2 chat template."""
        system = ROLE_PROMPTS[role]
        return (
            f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
            f"{user_message} [/INST]"
        )

    async def generate(
        self,
        user_message: str,
        role: str = "planner",
        request_id: str | None = None,
    ) -> InferenceResult:
        """Generate a single response for a given agent role."""
        if role not in ROLE_PROMPTS:
            raise ValueError(f"Unknown role: {role}. Choose from {list(ROLE_PROMPTS)}")

        rid = request_id or random_uuid()
        prompt = self._build_prompt(role, user_message)
        sampling_params = ROLE_SAMPLING[role]

        t0 = time.perf_counter()
        output_text = ""
        prompt_tokens = 0
        output_tokens = 0

        async for request_output in self.engine.generate(prompt, sampling_params, rid):
            if request_output.finished:
                output_text = request_output.outputs[0].text
                prompt_tokens = len(request_output.prompt_token_ids)
                output_tokens = len(request_output.outputs[0].token_ids)

        latency_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            role=role,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            text=output_text.strip(),
            latency_ms=round(latency_ms, 2),
            request_id=rid,
        )

    async def run_agent_pipeline(
        self,
        query: str,
        dataset_columns: list[str],
    ) -> dict:
        """
        Full Clarifier → Planner → Verifier pipeline for one query.
        All three turns share the SHARED_SYSTEM_PREFIX KV blocks.
        """
        results = {}

        # Turn 1: Clarifier
        clarifier_result = await self.generate(query, role="clarifier")
        results["clarifier"] = clarifier_result

        # Turn 2: Planner
        planner_input = f"Query: {query}\nDataset columns: {dataset_columns}"
        planner_result = await self.generate(planner_input, role="planner")
        results["planner"] = planner_result

        # Turn 3: Verifier (checks planner output against real columns)
        verifier_input = (
            f"Planner output: {planner_result.text}\n"
            f"Actual dataset columns: {dataset_columns}\n"
            f"Correct any hallucinated column names."
        )
        verifier_result = await self.generate(verifier_input, role="verifier")
        results["verifier"] = verifier_result

        total_ms = sum(r.latency_ms for r in results.values())
        results["total_latency_ms"] = round(total_ms, 2)
        results["total_tokens"] = sum(
            r.prompt_tokens + r.output_tokens for r in results.values()
        )
        return results

    async def run_concurrent_pipelines(
        self,
        queries: list[str],
        dataset_columns: list[str],
    ) -> list[dict]:
        """
        Run N agent pipelines concurrently.
        vLLM's continuous batching fills GPU compute across all concurrent requests.
        """
        tasks = [
            self.run_agent_pipeline(q, dataset_columns)
            for q in queries
        ]
        return await asyncio.gather(*tasks)


# ── OpenAI-compatible HTTP server ─────────────────────────────────────────────
def start_server(args):
    """
    Launch vLLM's built-in OpenAI-compatible server.
    AI-HOPE agents hit this at http://localhost:{port}/v1/completions
    """
    import subprocess, sys
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs),
        "--enable-prefix-caching",
        "--trust-remote-code",
    ]
    if args.tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]

    print(f"[server] Starting vLLM server on port {args.port}")
    print(f"[server] Command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    start_server(args)
