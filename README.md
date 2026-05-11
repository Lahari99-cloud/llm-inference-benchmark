# LLM Inference Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lahari99-cloud/llm-inference-benchmark/blob/main/notebooks/benchmark_analysis.ipynb)

Benchmarking framework comparing vLLM vs HuggingFace Transformers on latency, throughput, memory, and cost-per-token for production LLM inference at scale.

Built to answer the question every ML platform engineer faces: Which inference backend gives the best price-performance for our use case?

---

## Benchmark Results (Mistral-7B-Instruct-v0.3, A100 40GB)

| Framework | Avg Latency (ms) | Throughput (tok/s) | GPU Memory (GB) | Cost/1K tokens ($) |
|---|---|---|---|---|
| **vLLM** | **312** | **187.4** | 14.2 | **0.0021** |
| HuggingFace (fp16) | 891 | 64.3 | 15.8 | 0.0061 |
| HuggingFace (8-bit) | 1243 | 46.1 | 9.4 | 0.0087 |
| HuggingFace (4-bit) | 1087 | 52.8 | 6.1 | 0.0076 |

**Key Finding:** vLLM delivers **2.9x higher throughput** and **65% lower cost-per-token** vs vanilla HuggingFace fp16 on identical hardware (A100 40GB).

---

## Architecture

The benchmark harness runs identical prompt sets through multiple inference backends and collects metrics:

- **vLLM Runner** — PagedAttention async batched inference (GPU required)
- **HuggingFace Runner** — Baseline with fp16 / 8-bit / 4-bit BitsAndBytes quantization
- **TensorRT-LLM Runner** — Stub implementation (requires NVIDIA GPU server)
- **Metrics Collector** — TTFT, throughput, VRAM, and cost-per-token tracking
- **Analysis Layer** — Pandas + Plotly comparative visualization

---

## Tech Stack

- vLLM — High-throughput LLM serving with PagedAttention
- HuggingFace Transformers + BitsAndBytes — Baseline with quantization
- NVIDIA A100 / T4 — Benchmark hardware (Google Colab Pro+ compatible)
- Pandas + Plotly — Results analysis and visualization
- MLflow — Experiment tracking

---

## Quick Start

```bash
git clone https://github.com/Lahari99-cloud/llm-inference-benchmark
cd llm-inference-benchmark
pip install -r requirements.txt

# Run HuggingFace benchmark (CPU/GPU)
python benchmarks/hf_runner.py --model mistralai/Mistral-7B-Instruct-v0.3 --quantization 4bit

# Run vLLM benchmark (requires GPU)
python benchmarks/vllm_runner.py --model mistralai/Mistral-7B-Instruct-v0.3

# Analyze and plot results
python analysis/plot_results.py --results results/benchmark_results.json
```

---

## Repository Structure

```
llm-inference-benchmark/
├── benchmarks/
│   ├── base_runner.py            # Abstract base class for all runners
│   ├── vllm_runner.py            # vLLM async inference benchmarking
│   ├── hf_runner.py              # HuggingFace with fp16/8bit/4bit quant
│   └── tensorrt_runner.py        # TensorRT-LLM stub
├── analysis/
│   ├── plot_results.py           # Generate comparison charts
│   ├── cost_estimator.py         # Cost-per-token calculator
│   └── metrics_collector.py      # Latency, throughput, memory tracking
├── results/
│   └── benchmark_results_mistral7b.json
├── notebooks/
│   └── benchmark_analysis.ipynb  # Colab-ready interactive analysis
├── config/
│   └── benchmark_config.yaml     # Prompts, iterations, model configs
├── requirements.txt
└── README.md
```

---

## Key Metrics Measured

- **Time-to-First-Token (TTFT)** — Critical for streaming UX
- **Tokens/Second (Throughput)** — Batch inference capacity
- **GPU Memory (VRAM)** — Infrastructure cost driver
- **Cost per 1K tokens** — Based on A100 cloud pricing ($3.67/hr on AWS)
- **p50 / p95 / p99 Latency** — Tail latency for SLA planning

---

## Enterprise Context: Why Inference Optimization Matters

At scale, inference cost is the primary LLM deployment budget driver. For a system processing 10M transactions/day with LLM-augmented explainability:

| Scenario | Daily Cost (HF fp16) | Daily Cost (vLLM) | Annual Savings |
|---|---|---|---|
| 10M inferences/day | ~$610 | ~$210 | **~$146,000** |

This directly maps to Capital One's stated requirements around **scalability, latency, throughput, and cost optimization** for production AI systems.

---

## Reproducing Results

All benchmarks are reproducible on **Google Colab Pro+** (A100 runtime):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lahari99-cloud/llm-inference-benchmark/blob/main/notebooks/benchmark_analysis.ipynb)

---

Built by [Lahari Tadepalli](https://github.com/Lahari99-cloud) — AI Solutions Leader | 6+ years in production AI systems across finance, telecom, and enterprise operations.
