import json
import argparse
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_results(results_dir="results"):
    results = []
    for f in glob.glob(f"{results_dir}/*.json"):
        with open(f) as fp:
            data = json.load(fp)
            fw = data.get("framework", "unknown")
            q = data.get("quantization", "default")
            data["label"] = f"{fw} ({q})"
            results.append(data)
    return pd.DataFrame(results)


def plot_latency_comparison(df, output_path="results/latency_comparison.html"):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Latency (ms)", "p95 Latency (ms)"))
    colors = px.colors.qualitative.Set2[:len(df)]
    for i, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(name=row["label"], x=[row["label"]], y=[row["avg_latency_ms"]], marker_color=colors[i]), row=1, col=1)
        fig.add_trace(go.Bar(name=row["label"], x=[row["label"]], y=[row["p95_latency_ms"]], marker_color=colors[i], opacity=0.7), row=1, col=2)
    fig.update_layout(title="LLM Inference Latency Comparison", height=500)
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def plot_throughput_vs_cost(df, output_path="results/throughput_cost.html"):
    fig = px.scatter(
        df, x="throughput_tokens_per_sec", y="cost_per_1k_tokens_usd",
        text="label", size="gpu_memory_gb",
        title="Throughput vs Cost per 1K Tokens",
        color="label"
    )
    fig.update_traces(textposition="top center")
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def generate_summary_table(df):
    cols = ["label", "avg_latency_ms", "p95_latency_ms", "throughput_tokens_per_sec", "gpu_memory_gb", "cost_per_1k_tokens_usd"]
    subset = df[cols].copy()
    subset.columns = ["Framework", "Avg Latency (ms)", "p95 Latency (ms)", "Throughput (tok/s)", "GPU Mem (GB)", "Cost/1K ($)"]
    print("\n=== BENCHMARK SUMMARY ===")
    print(subset.to_markdown(index=False, floatfmt=".2f"))
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print("No results found. Run benchmarks first.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_latency_comparison(df, f"{args.output_dir}/latency_comparison.html")
    plot_throughput_vs_cost(df, f"{args.output_dir}/throughput_cost.html")
    generate_summary_table(df)


if __name__ == "__main__":
    main()
