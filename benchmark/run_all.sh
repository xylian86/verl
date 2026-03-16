#!/usr/bin/env bash
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
#
# Orchestration script for the multi-turn rollout sequence length benchmark.
#
# Usage:
#   cd /path/to/verl
#   bash benchmark/run_all.sh [--model MODEL] [--gpus N] [--skip-server] [--only USE_CASE]
#
# Prerequisites:
#   pip install vllm openai tiktoken pyyaml matplotlib datasets pandas
#
# This script:
#   1. Generates synthetic data (enterprise DB, prompts)
#   2. Downloads external datasets (SWE-bench, HotpotQA, etc.)
#   3. Starts a vLLM server (unless --skip-server)
#   4. Runs all 10 use case configs
#   5. Analyzes results and generates plots

set -euo pipefail

# --- Defaults ---
MODEL="Qwen/Qwen2.5-3B-Instruct"
GPUS=8
TP=1
PORT=8000
SKIP_SERVER=false
ONLY=""
MAX_SAMPLES=50
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --skip-server) SKIP_SERVER=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "Multi-Turn Rollout Sequence Length Benchmark"
echo "============================================"
echo "Model:       ${MODEL}"
echo "GPUs:        ${GPUS}"
echo "TP size:     ${TP}"
echo "Port:        ${PORT}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Skip server: ${SKIP_SERVER}"
echo "Only:        ${ONLY:-all}"
echo ""

cd "${PROJECT_ROOT}"

# --- Step 1: Generate synthetic data ---
echo ">>> Step 1: Generating synthetic data..."
python benchmark/data_gen/gen_enterprise_db.py \
    --db-path benchmark/data/enterprise.db \
    --nl2sql-output benchmark/data/nl2sql_prompts.jsonl \
    --pipeline-output benchmark/data/pipeline_debug_prompts.jsonl

python benchmark/data_gen/gen_synthetic_tasks.py \
    --gui-output benchmark/data/gui_tasks.jsonl \
    --research-output benchmark/data/research_tasks.jsonl \
    --codegen-output benchmark/data/codegen_tasks.jsonl \
    --n ${MAX_SAMPLES}

echo ""

# --- Step 2: Download external datasets ---
echo ">>> Step 2: Downloading external datasets..."
python benchmark/data_gen/download_datasets.py \
    --output-dir benchmark/data \
    --max-samples ${MAX_SAMPLES}

echo ""

# --- Step 3: Start vLLM server ---
VLLM_PID=""
if [ "${SKIP_SERVER}" = false ]; then
    echo ">>> Step 3: Starting vLLM server..."
    NUM_REPLICAS=$((GPUS / TP))
    echo "Starting vLLM with TP=${TP}, ${NUM_REPLICAS} replicas possible"

    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size ${TP} \
        --max-model-len 32768 \
        --port ${PORT} \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 64 &
    VLLM_PID=$!

    echo "Waiting for vLLM server to start (PID: ${VLLM_PID})..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            break
        fi
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "ERROR: vLLM server process died."
            exit 1
        fi
        sleep 2
    done

    if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: vLLM server failed to start within 4 minutes."
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
else
    echo ">>> Step 3: Skipping server start (--skip-server)"
fi

echo ""

# --- Step 4: Run benchmarks ---
echo ">>> Step 4: Running benchmarks..."
mkdir -p benchmark/results

BASE_URL="http://localhost:${PORT}/v1"

CONFIGS=(
    "swe_agent"
    "deep_research"
    "gui_computer_use"
    "multi_file_codegen"
    "agentgym_web"
    "verl_tool_search"
    "retool_code_interpreter"
    "swe_agent_recipe"
    "nl2sql_complex"
    "pipeline_debug"
)

for config_name in "${CONFIGS[@]}"; do
    if [ -n "${ONLY}" ] && [ "${config_name}" != "${ONLY}" ]; then
        continue
    fi

    echo ""
    echo "--- Running: ${config_name} ---"

    CONFIG_PATH="benchmark/configs/${config_name}.yaml"

    python -m benchmark.rollout_driver.multi_turn_driver \
        --config "${CONFIG_PATH}" \
        || echo "WARNING: ${config_name} failed, continuing..."
done

echo ""

# --- Step 5: Analyze results ---
echo ">>> Step 5: Analyzing results..."
python benchmark/analysis/plot_distributions.py \
    --results-dir benchmark/results \
    --output-dir benchmark/analysis_output

echo ""

# --- Cleanup ---
if [ -n "${VLLM_PID}" ]; then
    echo ">>> Shutting down vLLM server (PID: ${VLLM_PID})..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "Benchmark complete!"
echo "Results:  benchmark/results/"
echo "Analysis: benchmark/analysis_output/"
echo "============================================"
