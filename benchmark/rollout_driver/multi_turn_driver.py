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
Multi-turn rollout driver for measuring sequence length distributions.

Runs multi-turn agentic rollouts against a vLLM OpenAI-compatible server,
executing tool calls via pluggable tool executors. Records per-turn and
cumulative token counts to JSONL.
"""

import argparse
import asyncio
import importlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

import tiktoken
import yaml
from openai import AsyncOpenAI


@dataclass
class TurnRecord:
    turn: int
    role: str
    tokens: int
    cumulative_tokens: int
    finish_reason: str | None = None
    tool_calls_count: int = 0


@dataclass
class TrajectoryRecord:
    task_id: str
    use_case: str
    turns: list[TurnRecord] = field(default_factory=list)
    total_tokens: int = 0
    total_turns: int = 0
    finish_reason: str = ""  # "max_turns", "no_tool_call", "context_limit", "error"
    wall_time_s: float = 0.0


def count_tokens(text: str, encoding) -> int:
    """Count tokens using tiktoken encoding."""
    if text is None:
        return 0
    return len(encoding.encode(text, disallowed_special=()))


def count_message_tokens(messages: list[dict], encoding) -> int:
    """Count total tokens in a message list."""
    total = 0
    for msg in messages:
        total += 4  # message overhead
        content = msg.get("content") or ""
        total += count_tokens(content, encoding)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += count_tokens(json.dumps(tc.get("function", {})), encoding)
    return total


class MultiTurnDriver:
    def __init__(self, config: dict):
        self.config = config
        self.server_url = config["server"]["base_url"]
        self.model = config["server"]["model"]
        self.max_turns = config["rollout"]["max_turns"]
        self.max_tokens_per_turn = config["rollout"]["max_tokens_per_turn"]
        self.temperature = config["rollout"].get("temperature", 0.7)
        self.top_p = config["rollout"].get("top_p", 0.95)
        self.max_context_tokens = config["rollout"].get("max_context_tokens", 32000)
        self.concurrency = config.get("concurrency", 8)
        self.output_path = config["output"]["path"]

        # Load tool executor
        executor_cfg = config["tool_executor"]
        module = importlib.import_module(f"rollout_driver.tool_executors.{executor_cfg['module']}")
        executor_cls = getattr(module, executor_cfg["class"])
        self.executor = executor_cls(**executor_cfg.get("kwargs", {}))

        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Tools schema for the API
        self.tools = self.executor.get_tools_schema()

        self.client = AsyncOpenAI(base_url=self.server_url, api_key="token-abc123")

    async def run_single_trajectory(self, task: dict) -> TrajectoryRecord:
        """Run a single multi-turn trajectory."""
        task_id = task.get("task_id", "unknown")
        use_case = self.config["use_case"]
        record = TrajectoryRecord(task_id=task_id, use_case=use_case)

        messages = list(task["messages"])  # copy initial messages
        cumulative_tokens = count_message_tokens(messages, self.encoding)
        turn = 0
        start_time = time.time()

        try:
            while turn < self.max_turns:
                # Check context limit
                if cumulative_tokens >= self.max_context_tokens:
                    record.finish_reason = "context_limit"
                    break

                # Call LLM
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools if self.tools else None,
                        max_tokens=self.max_tokens_per_turn,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                except Exception as e:
                    record.finish_reason = f"error: {str(e)[:200]}"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                # Count assistant tokens
                assistant_content = assistant_msg.content or ""
                assistant_tokens = count_tokens(assistant_content, self.encoding)
                if assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        assistant_tokens += count_tokens(
                            json.dumps({"name": tc.function.name, "arguments": tc.function.arguments}),
                            self.encoding,
                        )

                cumulative_tokens += assistant_tokens
                n_tool_calls = len(assistant_msg.tool_calls) if assistant_msg.tool_calls else 0

                record.turns.append(
                    TurnRecord(
                        turn=turn,
                        role="assistant",
                        tokens=assistant_tokens,
                        cumulative_tokens=cumulative_tokens,
                        finish_reason=choice.finish_reason,
                        tool_calls_count=n_tool_calls,
                    )
                )

                # Build assistant message dict
                assistant_dict = {"role": "assistant", "content": assistant_content}
                if assistant_msg.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in assistant_msg.tool_calls
                    ]
                messages.append(assistant_dict)

                # If no tool calls, the model decided to stop
                if not assistant_msg.tool_calls:
                    record.finish_reason = "no_tool_call"
                    break

                # Execute tool calls
                for tc in assistant_msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_result = await self.executor.execute(tc.function.name, args)
                    tool_result_str = str(tool_result)

                    tool_tokens = count_tokens(tool_result_str, self.encoding)
                    cumulative_tokens += tool_tokens

                    record.turns.append(
                        TurnRecord(
                            turn=turn,
                            role="tool",
                            tokens=tool_tokens,
                            cumulative_tokens=cumulative_tokens,
                        )
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result_str,
                        }
                    )

                turn += 1

            if turn >= self.max_turns and not record.finish_reason:
                record.finish_reason = "max_turns"

        except Exception as e:
            record.finish_reason = f"error: {str(e)[:200]}"

        record.total_tokens = cumulative_tokens
        record.total_turns = turn
        record.wall_time_s = time.time() - start_time
        return record

    async def run_all(self, tasks: list[dict]) -> list[TrajectoryRecord]:
        """Run all tasks with bounded concurrency."""
        semaphore = asyncio.Semaphore(self.concurrency)
        results = []

        async def bounded_run(task):
            async with semaphore:
                return await self.run_single_trajectory(task)

        coros = [bounded_run(t) for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        records = []
        for r in results:
            if isinstance(r, Exception):
                records.append(
                    TrajectoryRecord(
                        task_id="error",
                        use_case=self.config["use_case"],
                        finish_reason=f"error: {str(r)[:200]}",
                    )
                )
            else:
                records.append(r)
        return records

    def save_results(self, records: list[TrajectoryRecord]):
        """Save results to JSONL."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            for rec in records:
                d = asdict(rec)
                f.write(json.dumps(d) + "\n")
        print(f"Saved {len(records)} trajectories to {self.output_path}")


def load_tasks(config: dict) -> list[dict]:
    """Load tasks from the dataset path specified in config."""
    data_path = config["dataset"]["path"]
    prompt_key = config["dataset"].get("prompt_key", "messages")
    task_id_key = config["dataset"].get("task_id_key", "task_id")
    max_samples = config["dataset"].get("max_samples", -1)
    system_prompt = config.get("system_prompt", None)

    tasks = []
    if data_path.endswith(".jsonl"):
        with open(data_path) as f:
            for line in f:
                tasks.append(json.loads(line))
    elif data_path.endswith(".json"):
        with open(data_path) as f:
            tasks = json.load(f)
    elif data_path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(data_path)
        tasks = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported dataset format: {data_path}")

    if max_samples > 0:
        tasks = tasks[:max_samples]

    # Normalize to {"task_id": ..., "messages": [...]}
    normalized = []
    for i, t in enumerate(tasks):
        messages = t.get(prompt_key, [])
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

        task_id = t.get(task_id_key, str(i))
        normalized.append({"task_id": task_id, "messages": messages})

    return normalized


async def main_async(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"=== Running use case: {config['use_case']} ===")
    print(f"Max turns: {config['rollout']['max_turns']}")
    print(f"Max tokens/turn: {config['rollout']['max_tokens_per_turn']}")
    print(f"Max context tokens: {config['rollout'].get('max_context_tokens', 32000)}")

    tasks = load_tasks(config)
    print(f"Loaded {len(tasks)} tasks")

    driver = MultiTurnDriver(config)
    records = await driver.run_all(tasks)
    driver.save_results(records)

    # Print summary
    total_tokens_list = [r.total_tokens for r in records]
    if total_tokens_list:
        import statistics

        print(f"\n--- Summary for {config['use_case']} ---")
        print(f"Tasks: {len(records)}")
        print(f"Mean tokens: {statistics.mean(total_tokens_list):.0f}")
        print(f"Median tokens: {statistics.median(total_tokens_list):.0f}")
        print(f"Max tokens: {max(total_tokens_list)}")
        print(f"Min tokens: {min(total_tokens_list)}")
        finish_reasons = {}
        for r in records:
            finish_reasons[r.finish_reason] = finish_reasons.get(r.finish_reason, 0) + 1
        print(f"Finish reasons: {finish_reasons}")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn rollout driver")
    parser.add_argument("--config", required=True, help="Path to use case YAML config")
    args = parser.parse_args()
    asyncio.run(main_async(args.config))


if __name__ == "__main__":
    main()
