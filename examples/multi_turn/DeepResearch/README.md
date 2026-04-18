# DeepResearch Search-R1 Example

This example adds a DeepResearch-style multi-turn GRPO recipe for Search-R1
data using async vLLM rollout.

It combines:

- the self-contained `deepresearcher` example layout from `zorro`
- the Search-R1 dataset and reward flow from `zorro/examples/multi_turn/search_r1`
- the retriever and preprocessing utilities already available in this repo

## Files

- `run_search_r1_grpo.sh`: main vLLM launch script (8-GPU defaults, remote retriever)
- `run_search_r1_grpo_local.sh`: 4-GPU quick-run launcher pinned to the repo-local
  `.verl` venv and a locally-running retriever
- `config/search_r1_grpo.yaml`: Hydra config for multi-turn rollout
- `config/tool_config/search_tool_config.yaml`: `search` tool definition
  (remote retriever URL)
- `config/tool_config/search_tool_config_local.yaml`: `search` tool pointing at
  a local retriever on `http://0.0.0.0:8001/retrieve`
- `config/reward_score.py`: Search-R1 exact-match reward wrapper

## Quick Start

1. Prepare Search-R1 parquet data:

```bash
python3 examples/data_preprocess/preprocess_search_datasets.py --dataset search_r1 \
  --local_dir ~/data/searchR1_processed_direct
```

2. Start a retrieval server.

You can reuse the local dense retriever under
`examples/sglang_multiturn/search_r1_like/local_dense_retriever/`, or point the
tool config at your own retrieval endpoint by editing
`config/tool_config/search_tool_config.yaml` (or the `_local` variant).

3. Launch training:

```bash
DATA_DIR=~/data/searchR1_processed_direct \
bash examples/multi_turn/DeepResearch/run_search_r1_grpo.sh
```

Common overrides (read from the environment by `run_search_r1_grpo.sh`):

```bash
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct   # default
N_GPUS=8
ROLLOUT_N=5
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
```

## Local quick-run (4 GPUs, local retriever)

For smoke-testing on a single node with GPUs 0-3 and a local retriever at
`http://0.0.0.0:8001/retrieve`:

```bash
bash examples/multi_turn/DeepResearch/run_search_r1_grpo_local.sh
```

This variant:

- uses the repo-local `.verl` virtual environment
  (`/code/users/xlian/verl_xlian/.verl`)
- pins `CUDA_VISIBLE_DEVICES=0,1,2,3` and `trainer.n_gpus_per_node=4`
- uses `config/tool_config/search_tool_config_local.yaml`
  (points at `http://0.0.0.0:8001/retrieve`)
- runs only 5 training steps with smaller batches and seq lengths, suitable for
  verifying the end-to-end pipeline
- defaults to `trainer.logger=['console']` so no `WANDB_API_KEY` is required

Override knobs via env vars, e.g.:

```bash
TRAIN_BATCH_SIZE=128 ROLLOUT_N=8 LOGGER="['console','wandb']" \
bash examples/multi_turn/DeepResearch/run_search_r1_grpo_local.sh
```

## Notes

- This recipe uses async vLLM multi-turn rollout and enables `VLLM_USE_V1=1`
  in the launch script.
- `preprocess_search_datasets.py` writes `train.parquet` and `test.parquet`,
  so the launch scripts expect those filenames under `DATA_DIR`.
- To change the retrieval endpoint, edit `retrieval_service_url` in the
  relevant `config/tool_config/search_tool_config*.yaml` file.
