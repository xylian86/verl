# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl is a flexible, efficient RL (Reinforcement Learning) training library for large language models. It supports algorithms like PPO, GRPO, DAPO, ReMax, RLOO, and more. It integrates with FSDP/FSDP2/Megatron-LM for training and vLLM/SGLang/HF Transformers for rollout generation. Orchestration is done via Ray.

## Common Commands

### Install (development)
```bash
pip install -e ".[test,vllm]"    # with vLLM backend
pip install -e ".[test,sglang]"  # with SGLang backend
```

### Linting and Formatting
Uses ruff via pre-commit. Line length is 120 chars.
```bash
pip install pre-commit
pre-commit install
pre-commit run                    # staged changes only
pre-commit run --all-files        # all files
pre-commit run --all-files ruff   # ruff only
```

Direct ruff usage:
```bash
ruff check --fix .                # lint + autofix
ruff format .                     # format
```

### Testing
Tests use pytest. Test directories are organized by execution environment:
- `tests/special_sanity/` — sanity checks (license, docstrings, compilation)
- `tests/special_standalone/` — standalone CPU/GPU tests
- `tests/special_distributed/` — multi-GPU distributed tests
- `tests/special_e2e/` — end-to-end training tests

Run a single test:
```bash
pytest tests/test_protocol_on_cpu.py -xvs
```

### Pre-commit Hooks
Beyond ruff, pre-commit also runs:
- `mypy` type checking
- `scripts/generate_trainer_config.sh` — auto-generates `verl/trainer/config/_generated_*.yaml` files from the dataclass configs; these must be committed
- `tests/special_sanity/check_docstrings.py` — enforces docstring coverage
- `tests/special_sanity/check_license.py` — enforces Apache 2.0 license headers on `examples/`, `scripts/`, `tests/`, `verl/`, and `setup.py`

## Architecture

### Core Data Protocol (`verl/protocol.py`)
`DataProto` is the universal data container passed between all components. It wraps a `TensorDict` with non-tensor metadata. All workers communicate using `DataProto`.

### Single Controller (`verl/single_controller/`)
The orchestration layer built on Ray. Provides `Worker`, `WorkerGroup`, and `ResourcePool` abstractions that manage distributed worker lifecycles. The `ray/` subdir contains the Ray-specific implementation.

### Trainer (`verl/trainer/`)
- `main_ppo.py` — Hydra entry point for PPO-family training (also used by GRPO, DAPO, etc.)
- `ppo/ray_trainer.py` — `RayPPOTrainer`, the main training loop orchestrator that coordinates actor, critic, reference, reward, and rollout workers
- `ppo/core_algos.py` — core RL algorithm implementations (advantage estimation, policy loss computation)
- `config/` — Hydra YAML configs + Python dataclass configs. `_generated_*.yaml` files are auto-generated from dataclasses; run `scripts/generate_trainer_config.sh` to regenerate
- `sft_trainer.py` — supervised fine-tuning trainer

### Workers (`verl/workers/`)
Each role in the RL pipeline has a worker implementation:
- `actor/` — policy model workers (`dp_actor.py` for FSDP, `megatron_actor.py` for Megatron)
- `critic/` — value model workers (same split)
- `rollout/` — generation backends (`vllm_rollout/`, `sglang_rollout/`, `hf_rollout.py`, `trtllm_rollout/`)
- `reward_manager/` — reward computation (model-based and function-based/verifiable rewards)
- `sharding_manager/` — handles model weight resharding between training and inference (3D-HybridEngine)
- `fsdp_workers.py` / `megatron_workers.py` — composite worker classes that bundle actor+critic+ref+reward roles

### Models (`verl/models/`)
Model-specific weight loading and parallelism support for Llama, Qwen2, and Megatron-Core models. `registry.py` and `weight_loader_registry.py` handle model registration.

### Configuration System
Uses Hydra with OmegaConf. Base configs live in `verl/trainer/config/`. Training is launched via Hydra entry points (e.g., `python -m verl.trainer.main_ppo`). Config overrides are passed as CLI args.

`verl/base_config.py` provides `BaseConfig`, a dataclass base that supports OmegaConf DictConfig-like access patterns.

### Experimental (`verl/experimental/`)
Features being incubated before merging into the main library: `fully_async_policy`, `one_step_off_policy`, `vla` (vision-language-action), `agent_loop`, `reward_loop`, `separation`, `dynamic_dataset`.

### Examples (`examples/`)
Each RL algorithm has its own directory (e.g., `ppo_trainer/`, `grpo_trainer/`) with shell scripts showing how to launch training with different models and configurations.

### Recipe (`recipe/`)
A git submodule pointing to [verl-recipe](https://github.com/verl-project/verl-recipe). Initialize with:
```bash
git submodule update --init --recursive recipe
```
