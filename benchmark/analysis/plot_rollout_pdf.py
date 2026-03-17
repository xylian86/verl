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
Plot rollout response length PDF distribution from JSONL benchmark results.

Usage:
    python benchmark/analysis/plot_rollout_pdf.py \
        --files benchmark/results/swe_agent.jsonl benchmark/results/swe_agent_thinking.jsonl \
        --labels "SWE-Agent" "SWE-Agent (Thinking)" \
        --output benchmark/analysis_output/rollout_pdf.pdf
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_tokens(filepath: str) -> np.ndarray:
    """Load total_tokens from a JSONL file."""
    tokens = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                tokens.append(r["total_tokens"])
    return np.array(tokens)


def plot_pdf(
    token_lists: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    output_path: str,
    max_length: int | None = None,
    bin_width: int = 100,
):
    """Plot PDF of rollout response lengths, matching the reference figure style."""
    from scipy.ndimage import gaussian_filter1d

    fig, ax = plt.subplots(figsize=(7, 4.5))

    all_tokens = np.concatenate(token_lists)
    x_max = max_length if max_length else int(np.max(all_tokens) * 1.05)
    bins = np.arange(0, x_max + bin_width, bin_width)

    for tokens, label, color in zip(token_lists, labels, colors):
        counts, edges = np.histogram(tokens, bins=bins)
        pdf = counts / len(tokens) / bin_width * 100 * bin_width  # PDF in %
        centers = (edges[:-1] + edges[1:]) / 2
        pdf_smooth = gaussian_filter1d(pdf, sigma=3)
        ax.plot(centers, pdf_smooth, color=color, linewidth=1.8, label=label)

    if max_length:
        ax.axvline(
            x=max_length, color="gray", linestyle="--", linewidth=1.2, zorder=0
        )
        ax.annotate(
            f"Max Length: {max_length // 1000}K",
            xy=(max_length, ax.get_ylim()[1] * 0.55),
            xytext=(-15, 15),
            textcoords="offset points",
            fontsize=10,
            ha="right",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
        )

    ax.set_xlabel("Rollout Response Length (tokens)", fontsize=16)
    ax.set_ylabel("PDF (%)", fontsize=16)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x // 1000)}K" if x >= 1000 else f"{int(x)}"
    ))

    ax.tick_params(axis="both", labelsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved PDF plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot rollout response length PDF distribution"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="JSONL result files to plot",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Legend labels (one per file). Defaults to filenames.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="Line colors (one per file).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Draw a vertical dashed line at this max length (e.g. 30000).",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=100,
        help="Histogram bin width in tokens (default: 100).",
    )
    parser.add_argument(
        "--output",
        default="benchmark/analysis_output/rollout_pdf.pdf",
        help="Output file path.",
    )
    args = parser.parse_args()

    token_lists = [load_tokens(f) for f in args.files]

    if args.labels:
        labels = args.labels
    else:
        labels = [f.rsplit("/", 1)[-1].replace(".jsonl", "") for f in args.files]

    default_colors = ["#8B1A1A", "#1E90FF", "#2E8B57", "#FF8C00", "#6A0DAD"]
    colors = args.colors if args.colors else default_colors[: len(token_lists)]

    for f, label, tokens in zip(args.files, labels, token_lists):
        print(
            f"  {label}: {len(tokens)} samples, "
            f"mean={np.mean(tokens):.0f}, median={np.median(tokens):.0f}, "
            f"max={np.max(tokens)}"
        )

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    plot_pdf(token_lists, labels, colors, args.output, args.max_length, args.bin_width)


if __name__ == "__main__":
    main()
