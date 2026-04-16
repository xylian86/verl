#!/bin/bash

set -euo pipefail

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}

# Freeze the currently installed packages so later installs fail instead of
# silently upgrading or downgrading existing versions.
CONSTRAINTS_FILE=$(mktemp)
cleanup() {
    rm -f "$CONSTRAINTS_FILE"
}
trap cleanup EXIT

refresh_constraints() {
    uv pip freeze --python "$PYTHON_BIN" --exclude-editable \
        | awk '!/^(datasets|fsspec|packaging)==/' > "$CONSTRAINTS_FILE"
}

pip_install_preserve_versions() {
    uv pip install --python "$PYTHON_BIN" --no-cache -c "$CONSTRAINTS_FILE" "$@"
    refresh_constraints
}

refresh_constraints

echo "1. install inference frameworks and pytorch they need"

echo "2. install basic packages"
pip_install_preserve_versions "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

pip_install_preserve_versions "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
pip_install_preserve_versions flashinfer-python==0.3.1


echo "5. May need to fix opencv"
pip_install_preserve_versions opencv-python
pip_install_preserve_versions opencv-fixer
echo "Skipping automatic opencv-fixer run to avoid changing existing package versions."
echo "Run it manually if needed: python -c \"from opencv_fixer import AutoFix; AutoFix()\""


echo "Successfully installed all packages"
