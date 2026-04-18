# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
"""Unified preprocessor for the multi-turn DeepResearch training pipeline.

Supports two source datasets and writes them into the same verl-compatible
parquet schema (see `process_single_row` below). Running it a second time
with a different ``--dataset`` choice produces another set of parquet files
under a different local directory; both can coexist and you simply point
``data.train_files`` / ``data.val_files`` at the one you want to train on.

Supported datasets:
  * ``search_r1`` (default) - ``PeterJinGo/nq_hotpotqa_train`` (NQ + HotpotQA).
    The HF repo already has ``train.parquet`` and ``test.parquet``.
  * ``deepresearch_9k`` - ``artillerywu/DeepResearch-9K``. Single train split
    with 3,974 multi-hop questions; we hold out ``--val_size`` rows at the
    tail as the test split.

Usage:
    python preprocess_search_datasets.py --dataset search_r1
    python preprocess_search_datasets.py --dataset deepresearch_9k --val_size 200
"""

import argparse
import logging
import os
import tempfile
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The chat-template inputs below are shared across datasets so the Hermes
# tool-call parser, tool config, and reward function all keep working
# unchanged when switching between sources.
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "every time before you act. After reasoning, if you lack knowledge, call the "
    "`search` tool by emitting a tool call in this EXACT format (a single JSON object "
    "wrapped in <tool_call>...</tool_call>):\n"
    "<tool_call>\n"
    '{"name": "search", "arguments": {"query_list": ["your first query", "optional second query"]}}\n'
    "</tool_call>\n"
    "The retrieved evidence will be returned to you in a tool message wrapped in "
    "<tool_response> and </tool_response>. You may call `search` as many times as you "
    "need. When you have enough information, give the final answer inside <answer> and "
    "</answer>, without detailed illustrations. For example, <answer> Beijing "
    "</answer>. Question: "
)


# --- Dataset registry -------------------------------------------------------
# Each entry captures the HF repo id, its default local output directory, and
# a callable that maps a raw row dict -> (question, ground_truth_list,
# data_source_tag, ability, metadata). Everything else (prompt building, tool
# kwargs, reward_model block) is shared across datasets.


def _map_search_r1_row(row: dict[str, Any]) -> tuple[str, Any, str, Any, Any, Any]:
    """Map a Search-R1 (PeterJinGo/nq_hotpotqa_train) row.

    Preserves prior behavior: ``reward_model`` dict is kept as-is when
    present (it already contains ``ground_truth.target``); falls back to
    ``golden_answers`` for older snapshots.
    """
    question = row.get("question", "")

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    data_source_tag = "searchR1_" + str(row.get("data_source", ""))
    ability = row.get("ability")
    metadata = row.get("metadata")
    return question, ground_truth, data_source_tag, ability, metadata, reward_model_data


def _map_deepresearch_9k_row(row: dict[str, Any]) -> tuple[str, Any, str, Any, Any, Any]:
    """Map an artillerywu/DeepResearch-9K row.

    The raw row has ``question``, ``difficulty`` (1-3), ``search trajectory``
    (a reference chain we don't need for GRPO), and ``final answer``. We wrap
    the single ``final answer`` in a one-element list so the shared EM reward
    (``ground_truth['target']``) can iterate over it uniformly.
    """
    question = row.get("question", "") or ""
    final_answer = row.get("final answer")
    if final_answer is None:
        final_answer = ""
    ground_truth = {"target": [str(final_answer)]}
    reward_model_data = {"style": "rule", "ground_truth": ground_truth}

    difficulty = row.get("difficulty")
    try:
        difficulty_int = int(difficulty) if difficulty is not None else None
    except (TypeError, ValueError):
        difficulty_int = None

    data_source_tag = "deepresearch_9k"
    ability = f"difficulty_{difficulty_int}" if difficulty_int is not None else "deepresearch_9k"
    # Keep only small structured metadata; the raw ``search trajectory`` can
    # be tens of kilobytes per row and we don't need it for RL rollouts.
    metadata = {"difficulty": difficulty_int}
    return question, ground_truth, data_source_tag, ability, metadata, reward_model_data


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "search_r1": {
        "hf_repo_id": "PeterJinGo/nq_hotpotqa_train",
        "default_local_dir": "~/data/searchR1_processed_direct",
        # Map split name -> filename inside the HF repo.
        "hf_filenames": {"train": "train.parquet", "test": "test.parquet"},
        "val_split_from_train": None,
        "row_mapper": _map_search_r1_row,
    },
    "deepresearch_9k": {
        "hf_repo_id": "artillerywu/DeepResearch-9K",
        "default_local_dir": "~/data/deepresearch_9k_processed",
        # The HF repo groups its single train split under a subfolder.
        "hf_filenames": {"train": "DeepResearch-9K/train-00000-of-00001.parquet"},
        # Take the last N rows of train as the held-out test set.
        "val_split_from_train": "tail",
        "row_mapper": _map_deepresearch_9k_row,
    },
}


def process_single_row(
    row: dict[str, Any],
    current_split_name: str,
    row_index: int,
    row_mapper,
    system_content: str,
    user_content_prefix: str,
) -> pd.Series:
    """Build a verl-compatible row from any supported dataset.

    Returns a Series with these columns (matching the Search-R1 layout):
      data_source, prompt, ability, reward_model, extra_info, metadata.
    """
    question, ground_truth, data_source_tag, ability, metadata, reward_model_data = row_mapper(row)

    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source_tag,
            }
        }
    }

    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tag,
            "prompt": prompt,
            "ability": ability,
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": metadata,
        }
    )


def _download_split(hf_repo_id: str, filename: str, tmp_dir: str) -> str:
    logger.info(f"Downloading {filename} from {hf_repo_id}")
    return hf_hub_download(
        repo_id=hf_repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
    )


def _process_and_save(
    df_raw: pd.DataFrame,
    split_name: str,
    out_path: str,
    row_mapper,
    system_content: str,
    user_content_prefix: str,
) -> int:
    logger.info(f"Processing {len(df_raw)} rows for split '{split_name}'")

    def _apply(row: pd.Series) -> pd.Series:
        return process_single_row(
            row=row,
            current_split_name=split_name,
            row_index=row.name,
            row_mapper=row_mapper,
            system_content=system_content,
            user_content_prefix=user_content_prefix,
        )

    df_processed = df_raw.apply(_apply, axis=1)
    df_processed.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df_processed)} rows -> {out_path}")
    return len(df_processed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download one of the supported Search / DeepResearch datasets from "
            "Hugging Face, convert it to the verl multi-turn parquet schema, "
            "and save locally."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY.keys()),
        default="search_r1",
        help="Which dataset to preprocess.",
    )
    parser.add_argument(
        "--hf_repo_id",
        default=None,
        help="Override the HF repo id. Defaults to the one registered for --dataset.",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help=(
            "Local directory for the output parquet files. Defaults to the "
            "per-dataset location registered in DATASET_REGISTRY."
        ),
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=200,
        help=(
            "Number of rows to hold out as the test split when the source "
            "only provides a single train split. Ignored when the HF repo "
            "already ships a test split."
        ),
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the parquet files to.",
    )
    args = parser.parse_args()

    cfg = DATASET_REGISTRY[args.dataset]
    hf_repo_id = args.hf_repo_id or cfg["hf_repo_id"]
    local_dir = os.path.expanduser(args.local_dir or cfg["default_local_dir"])
    os.makedirs(local_dir, exist_ok=True)

    row_mapper = cfg["row_mapper"]
    hf_filenames: dict[str, str] = dict(cfg["hf_filenames"])
    splits = list(hf_filenames.keys())
    val_strategy = cfg["val_split_from_train"]

    logger.info(
        f"Preprocessing dataset '{args.dataset}' from {hf_repo_id} -> {local_dir} "
        f"(splits on HF: {splits}; synthetic val strategy: {val_strategy})"
    )

    processed_files: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        # --- Case 1: HF repo already has train + test files. ---
        if "test" in splits:
            for split in splits:
                try:
                    local_parquet = _download_split(
                        hf_repo_id, hf_filenames[split], tmp_download_dir
                    )
                    df_raw = pd.read_parquet(local_parquet)
                    out_path = os.path.join(local_dir, f"{split}.parquet")
                    _process_and_save(
                        df_raw=df_raw,
                        split_name=split,
                        out_path=out_path,
                        row_mapper=row_mapper,
                        system_content=DEFAULT_SYSTEM_CONTENT,
                        user_content_prefix=DEFAULT_USER_CONTENT_PREFIX,
                    )
                    processed_files.append(out_path)
                except EntryNotFoundError:
                    logger.warning(
                        f"{hf_filenames[split]} not found in {hf_repo_id}; skipping."
                    )
                except Exception as e:
                    logger.error(f"Error processing split '{split}': {e}")
        else:
            # --- Case 2: only 'train' exists; synthesize a val split. ---
            if "train" not in splits:
                raise ValueError(f"DATASET_REGISTRY for {args.dataset!r} must include a 'train' split.")
            local_parquet = _download_split(hf_repo_id, hf_filenames["train"], tmp_download_dir)
            df_raw = pd.read_parquet(local_parquet)
            n_total = len(df_raw)
            val_size = max(0, min(args.val_size, n_total))
            logger.info(f"Source has {n_total} rows; holding out last {val_size} as test.")

            if val_strategy != "tail":
                raise ValueError(
                    f"Unsupported val_split_from_train strategy: {val_strategy!r}. "
                    "Only 'tail' is implemented."
                )

            if val_size > 0:
                df_train = df_raw.iloc[:-val_size].reset_index(drop=True)
                df_test = df_raw.iloc[-val_size:].reset_index(drop=True)
            else:
                df_train = df_raw.reset_index(drop=True)
                df_test = df_raw.iloc[:0].reset_index(drop=True)

            for split_name, df_split in (("train", df_train), ("test", df_test)):
                if len(df_split) == 0:
                    logger.warning(f"Skipping empty split '{split_name}'.")
                    continue
                out_path = os.path.join(local_dir, f"{split_name}.parquet")
                _process_and_save(
                    df_raw=df_split,
                    split_name=split_name,
                    out_path=out_path,
                    row_mapper=row_mapper,
                    system_content=DEFAULT_SYSTEM_CONTENT,
                    user_content_prefix=DEFAULT_USER_CONTENT_PREFIX,
                )
                processed_files.append(out_path)

    if not processed_files:
        logger.warning("No data was processed or saved.")
        return

    logger.info(f"Successfully processed {len(processed_files)} file(s) -> {local_dir}")

    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            logger.info(f"Copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    main()
