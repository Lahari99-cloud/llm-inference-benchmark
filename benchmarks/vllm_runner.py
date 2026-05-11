"""
vLLM Async Inference Benchmark Runner
High-throughput inference using PagedAttention
"""

import time
import json
import argparse
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    framework: str
    model: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float
    gpu_memory_gb: float
    cost_per_1k_tokens_usd: float
    num_samples: int


class VLLMRunner:
    """
    Benchmarks vLLM inference engine with PagedAttention.
    vLLM provides 2-4x higher throughput vs vanilla HuggingFace
    through continuous batching and memory-efficient KV cache management.
    """

    GPU_HOURLY_COST_USD = 3.67  # A100 40GB on AWS on-demand

    def __init__(self, model_name, gpu_memory_utilization=0.90, max_model_len=4096):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None

    def load_model(self):
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
            self.SamplingParams = SamplingParams
            logger.info(f"vLLM model loaded: {self.model_name}")
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

    def benchmark(self, prompts, max_tokens=256, warmup_runs=3):
        if not self.llm:
            self.load_model()

        sampling_params = self.SamplingParams(temperature=0, max_tokens=max_tokens)

        # Warmup
        logger.info(f"Warming up with {warmup_runs} iterations...")
        self.llm.generate(prompts[:warmup_runs], sampling_params)

        # Benchmark - vLLM handles batching internally
        latencies = []
        total_tokens = 0
        start_total = time.perf_counter()

        for prompt in tqdm(prompts, desc="Benchmarking"):
            start = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            total_tokens += len(outputs[0].outputs[0].token_ids)

        total_time_s = time.perf_counter() - start_total
        throughput = total_tokens / total_time_s
        cost_per_1k = (self.GPU_HOURLY_COST_USD / 3600) * (1000 / max(throughput, 1))

        try:
            import torch
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            gpu_mem = 0.0

        return BenchmarkResult(
            framework="vllm",
            model=self.model_name,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_tokens_per_sec=throughput,
            gpu_memory_gb=gpu_mem,
            cost_per_1k_tokens_usd=cost_per_1k,
            num_samples=len(prompts)
        )


def main():
    parser = argparse.ArgumentParser(description="vLLM Inference Benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="results/vllm_results.json")
    args = parser.parse_args()

    prompts = [
        "Explain the concept of PagedAttention and why it improves LLM throughput.",
        "What are best practices for production LLM deployment at scale?",
        "Compare different quantization strategies for LLM inference.",
        "How does continuous batching improve GPU utilization in vLLM?",
        "Describe the architecture of a high-throughput LLM serving system.",
    ] * (args.num_samples // 5 + 1)
    prompts = prompts[:args.num_samples]

    runner = VLLMRunner(model_name=args.model)
    result = runner.benchmark(prompts, max_tokens=args.max_tokens)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"vLLM Results: {result.avg_latency_ms:.1f}ms avg | {result.throughput_tokens_per_sec:.1f} tok/s")


if __name__ == "__main__":
    main()
