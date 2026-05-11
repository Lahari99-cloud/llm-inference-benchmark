"""
HuggingFace Transformers Inference Benchmark Runner
Supports fp16, 8-bit (bitsandbytes), and 4-bit quantization
"""

import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    framework: str
    model: str
    quantization: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float
    gpu_memory_gb: float
    cost_per_1k_tokens_usd: float
    num_samples: int


class HuggingFaceRunner:
    """
    Benchmarks HuggingFace Transformers inference with multiple quantization configs.
    Measures TTFT, throughput, memory usage, and cost-per-token.
    """
    
    GPU_HOURLY_COST_USD = 3.67  # A100 40GB on AWS on-demand
    
    def __init__(self, model_name, quantization="fp16", device="auto", max_new_tokens=256):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
    
    def _get_bnb_config(self):
        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None
    
    def load_model(self):
        logger.info(f"Loading {self.model_name} with {self.quantization} quantization...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        bnb_config = self._get_bnb_config()
        model_kwargs = {
            "torch_dtype": torch.float16 if self.quantization == "fp16" else None,
            "device_map": self.device,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
    
    def _get_gpu_memory_gb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0
    
    def _run_single_inference(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        latency_ms = (time.perf_counter() - start) * 1000
        output_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
        return latency_ms, output_tokens
    
    def benchmark(self, prompts, warmup_runs=3):
        if not self.model:
            self.load_model()
        for _ in range(min(warmup_runs, len(prompts))):
            self._run_single_inference(prompts[0])
        latencies = []
        total_tokens = 0
        start_total = time.perf_counter()
        for prompt in tqdm(prompts):
            latency_ms, tokens = self._run_single_inference(prompt)
            latencies.append(latency_ms)
            total_tokens += tokens
        total_time_s = time.perf_counter() - start_total
        throughput = total_tokens / total_time_s
        cost_per_1k = (self.GPU_HOURLY_COST_USD / 3600) * (1000 / max(throughput, 1))
        return BenchmarkResult(
            framework="huggingface",
            model=self.model_name,
            quantization=self.quantization,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_tokens_per_sec=throughput,
            gpu_memory_gb=self._get_gpu_memory_gb(),
            cost_per_1k_tokens_usd=cost_per_1k,
            num_samples=len(prompts)
        )


def main():
    parser = argparse.ArgumentParser(description="HuggingFace LLM Inference Benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--quantization", choices=["fp16", "8bit", "4bit"], default="fp16")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="results/hf_results.json")
    args = parser.parse_args()
    
    prompts = [
        "Explain PagedAttention in LLM inference systems.",
        "What are the differences between RAG and fine-tuning?",
        "Describe a multi-agent architecture for fraud detection.",
        "What is the role of FAISS in vector similarity search?",
        "Explain how transformer attention scales with sequence length.",
    ] * (args.num_samples // 5 + 1)
    prompts = prompts[:args.num_samples]
    
    runner = HuggingFaceRunner(
        model_name=args.model,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens
    )
    result = runner.benchmark(prompts)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Avg Latency: {result.avg_latency_ms:.1f}ms | Throughput: {result.throughput_tokens_per_sec:.1f} tok/s")


if __name__ == "__main__":
    main()
