#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.verl"
PYTHON_BIN="$VENV_DIR/bin/python"
DATA_ROOT="$HOME/data"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Expected Python environment at $PYTHON_BIN" >&2
    echo "Create it with: uv venv \"$VENV_DIR\" --python 3.12" >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

gsm8k_dir="$DATA_ROOT/gsm8k"
math_dir="$DATA_ROOT/math"
gsm8k_train_path="$gsm8k_dir/train.parquet"
gsm8k_test_path="$gsm8k_dir/test.parquet"
math_train_path="$math_dir/train.parquet"
math_test_path="$math_dir/test.parquet"

ensure_dataset() {
    local dataset_name="$1"
    local train_path="$2"
    local test_path="$3"
    local preprocess_script="$4"
    local save_dir="$5"

    if [ -s "$train_path" ] && [ -s "$test_path" ]; then
        echo "$dataset_name dataset already exists in $save_dir"
        return
    fi

    echo "$dataset_name dataset not found. Downloading and preprocessing into $save_dir"
    mkdir -p "$save_dir"
    "$PYTHON_BIN" "$preprocess_script" --local_save_dir "$save_dir"
}

ensure_dataset "GSM8K" "$gsm8k_train_path" "$gsm8k_test_path" "$REPO_ROOT/examples/data_preprocess/gsm8k.py" "$gsm8k_dir"
ensure_dataset "MATH" "$math_train_path" "$math_test_path" "$REPO_ROOT/examples/data_preprocess/math_dataset.py" "$math_dir"

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=20 \
    trainer.total_epochs=1 \
    "$@"
