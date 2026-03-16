# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Analyze rollout results and plot sequence length distributions.

Reads JSONL files from benchmark/results/ and produces:
- Per-use-case histograms
- Combined CDF plot
- Box plots comparing use cases
- Summary statistics table
"""

import argparse
import json
import os
import statistics
from collections import defaultdict


def load_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all JSONL result files from a directory."""
    all_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".jsonl"):
            continue
        use_case = fname.replace(".jsonl", "")
        records = []
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if records:
            all_results[use_case] = records
    return all_results


def compute_stats(records: list[dict]) -> dict:
    """Compute summary statistics for a use case."""
    tokens = [r["total_tokens"] for r in records]
    turns = [r["total_turns"] for r in records]
    wall_times = [r.get("wall_time_s", 0) for r in records]

    finish_reasons = defaultdict(int)
    for r in records:
        finish_reasons[r["finish_reason"]] += 1

    stats = {
        "count": len(records),
        "tokens_mean": statistics.mean(tokens) if tokens else 0,
        "tokens_median": statistics.median(tokens) if tokens else 0,
        "tokens_stdev": statistics.stdev(tokens) if len(tokens) > 1 else 0,
        "tokens_min": min(tokens) if tokens else 0,
        "tokens_max": max(tokens) if tokens else 0,
        "tokens_p90": sorted(tokens)[int(0.9 * len(tokens))] if tokens else 0,
        "tokens_p95": sorted(tokens)[int(0.95 * len(tokens))] if tokens else 0,
        "tokens_p99": sorted(tokens)[min(int(0.99 * len(tokens)), len(tokens) - 1)] if tokens else 0,
        "turns_mean": statistics.mean(turns) if turns else 0,
        "turns_median": statistics.median(turns) if turns else 0,
        "turns_max": max(turns) if turns else 0,
        "wall_time_mean": statistics.mean(wall_times) if wall_times else 0,
        "finish_reasons": dict(finish_reasons),
    }
    return stats


def print_summary_table(all_results: dict[str, list[dict]]):
    """Print a summary table to stdout."""
    print("\n" + "=" * 120)
    print(f"{'Use Case':<25} {'Count':>6} {'Mean Tok':>10} {'Med Tok':>10} {'P90 Tok':>10} {'P95 Tok':>10} {'Max Tok':>10} {'Mean Turns':>11} {'Max Turns':>10}")
    print("=" * 120)

    for use_case, records in sorted(all_results.items()):
        s = compute_stats(records)
        print(
            f"{use_case:<25} {s['count']:>6} {s['tokens_mean']:>10.0f} {s['tokens_median']:>10.0f} "
            f"{s['tokens_p90']:>10} {s['tokens_p95']:>10} {s['tokens_max']:>10} "
            f"{s['turns_mean']:>11.1f} {s['turns_max']:>10}"
        )

    print("=" * 120)

    # Finish reason breakdown
    print("\nFinish Reason Breakdown:")
    print("-" * 80)
    for use_case, records in sorted(all_results.items()):
        s = compute_stats(records)
        reasons = ", ".join(f"{k}: {v}" for k, v in sorted(s["finish_reasons"].items()))
        print(f"  {use_case:<25} {reasons}")


def save_stats_json(all_results: dict[str, list[dict]], output_path: str):
    """Save detailed statistics to JSON."""
    all_stats = {}
    for use_case, records in sorted(all_results.items()):
        all_stats[use_case] = compute_stats(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats to {output_path}")


def plot_distributions(all_results: dict[str, list[dict]], output_dir: str):
    """Generate plots. Requires matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping plots. Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # 1. Per-use-case histograms
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    for idx, (use_case, records) in enumerate(sorted(all_results.items())):
        if idx >= len(axes):
            break
        tokens = [r["total_tokens"] for r in records]
        ax = axes[idx]
        ax.hist(tokens, bins=30, color=colors[idx], alpha=0.7, edgecolor="black")
        ax.set_title(use_case.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Total Tokens")
        ax.set_ylabel("Count")
        ax.axvline(statistics.median(tokens) if tokens else 0, color="red", linestyle="--", label="median")
        ax.legend(fontsize=8)
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("Sequence Length Distribution per Use Case", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograms.png"), dpi=150)
    plt.close()
    print(f"Saved histograms to {output_dir}/histograms.png")

    # 2. Combined CDF
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, (use_case, records) in enumerate(sorted(all_results.items())):
        tokens = sorted([r["total_tokens"] for r in records])
        if not tokens:
            continue
        cdf = np.arange(1, len(tokens) + 1) / len(tokens)
        ax.plot(tokens, cdf, label=use_case, color=colors[idx], linewidth=2)
    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Cumulative Distribution of Sequence Lengths", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf.png"), dpi=150)
    plt.close()
    print(f"Saved CDF to {output_dir}/cdf.png")

    # 3. Box plots
    fig, ax = plt.subplots(figsize=(14, 8))
    data = []
    labels = []
    for use_case, records in sorted(all_results.items()):
        tokens = [r["total_tokens"] for r in records]
        if tokens:
            data.append(tokens)
            labels.append(use_case.replace("_", "\n"))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Total Tokens", fontsize=12)
    ax.set_title("Sequence Length Distribution Comparison", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot.png"), dpi=150)
    plt.close()
    print(f"Saved boxplot to {output_dir}/boxplot.png")

    # 4. Tokens vs Turns scatter
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, (use_case, records) in enumerate(sorted(all_results.items())):
        tokens = [r["total_tokens"] for r in records]
        turns = [r["total_turns"] for r in records]
        ax.scatter(turns, tokens, label=use_case, color=colors[idx], alpha=0.6, s=30)
    ax.set_xlabel("Number of Turns", fontsize=12)
    ax.set_ylabel("Total Tokens", fontsize=12)
    ax.set_title("Tokens vs Turns by Use Case", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_tokens_vs_turns.png"), dpi=150)
    plt.close()
    print(f"Saved scatter plot to {output_dir}/scatter_tokens_vs_turns.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout sequence length distributions")
    parser.add_argument("--results-dir", default="benchmark/results", help="Directory containing result JSONL files")
    parser.add_argument("--output-dir", default="benchmark/analysis_output", help="Directory for plots and stats")
    args = parser.parse_args()

    all_results = load_results(args.results_dir)
    if not all_results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Loaded results for {len(all_results)} use cases")
    print_summary_table(all_results)
    save_stats_json(all_results, os.path.join(args.output_dir, "summary_stats.json"))
    plot_distributions(all_results, args.output_dir)


if __name__ == "__main__":
    main()
