#!/usr/bin/env bash
# =============================================================================
# Local Search-R1 GRPO on Qwen2.5-3B-Instruct using vLLM rollout.
#
# Uses:
#   * repo-local .verl venv
#   * GPUs 0-3 only
#   * local retrieval server at http://0.0.0.0:8001/retrieve
# =============================================================================
set -euxo pipefail

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_DIR="$REPO_ROOT/.verl"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Expected Python environment at $PYTHON_BIN" >&2
    exit 1
fi

export VIRTUAL_ENV="/code/users/xlian/verl_xlian/.verl"
export PATH="/code/users/xlian/verl_xlian/.verl/bin:$PATH"
export PYTHONPATH="/code/users/xlian/verl_xlian"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_V1="1"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export HYDRA_FULL_ERROR=1

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-$HOME/data/searchR1_processed_direct}"
N_GPUS="${N_GPUS:-4}"
PROJECT_NAME="${PROJECT_NAME:-deepresearch-search-r1-local}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepresearch-search-r1-local-$(date +%Y%m%d-%H%M)}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
ROLLOUT_N="${ROLLOUT_N:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"

# Preserve wandb setting; default to console-only to avoid surprise failures
# when WANDB_API_KEY is not set. Override by setting LOGGER='["console","wandb"]'.
LOGGER="${LOGGER:-['console']}"

CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="${TOOL_CONFIG:-$CONFIG_PATH/tool_config/search_tool_config_local.yaml}"
REWARD_FN="$CONFIG_PATH/reward_score.py"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

if [ ! -s "$DATA_DIR/train.parquet" ] || [ ! -s "$DATA_DIR/test.parquet" ]; then
    echo "Dataset missing at $DATA_DIR; run preprocess_search_r1_dataset.py first." >&2
    exit 1
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="search_r1_grpo" \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size=64 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    "$@" \
    2>&1 | tee "$LOG_FILE"
