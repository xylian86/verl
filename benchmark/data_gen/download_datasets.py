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
Download and prepare external datasets for benchmark use cases.

Downloads:
- SWE-bench Lite (from HuggingFace)
- HotpotQA (dev set)
- BIRD-SQL (dev set)
- CodeContests (from HuggingFace, small subset)
- NuminaMath-TIR (from HuggingFace, small subset)
- AgentGym web tasks (synthetic if not available)

Converts all to a unified JSONL format: {"task_id": ..., "messages": [...]}
"""

import argparse
import json
import os


def download_swe_bench(output_path: str, max_samples: int = 50):
    """Download SWE-bench Lite and convert to JSONL."""
    try:
        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except Exception as e:
        print(f"Could not download SWE-bench Lite: {e}")
        print("Generating synthetic SWE-bench-style prompts instead.")
        _gen_synthetic_swe(output_path, max_samples)
        return

    tasks = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        system_prompt = (
            "You are an autonomous software engineering agent. You have access to a bash shell. "
            "Your task is to resolve the following GitHub issue by finding the relevant code, "
            "understanding the bug, and writing a fix. Use bash commands to explore the repository, "
            "read files, and make changes. When done, show the final diff."
        )

        issue_text = f"Repository: {row['repo']}\n\nIssue: {row['problem_statement']}"
        if row.get("hints_text"):
            issue_text += f"\n\nHints: {row['hints_text']}"

        tasks.append({
            "task_id": row.get("instance_id", f"swe_{i:03d}"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": issue_text},
            ],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} SWE-bench tasks to {output_path}")


def _gen_synthetic_swe(output_path: str, n: int):
    """Fallback: generate synthetic SWE-agent style tasks."""
    issues = [
        ("django/django", "QuerySet.bulk_create() crashes with unique constraint violation on PostgreSQL when ignore_conflicts=True and update_conflicts=True are both set."),
        ("psf/requests", "Session.send() doesn't properly handle chunked transfer encoding when stream=True and the server sends trailers."),
        ("pandas-dev/pandas", "DataFrame.groupby().rolling() produces incorrect results when the groupby column has NaN values and the window is time-based."),
        ("scikit-learn/scikit-learn", "HistGradientBoostingClassifier.predict_proba() returns inconsistent results when n_classes=2 and the model was trained with class_weight='balanced'."),
        ("matplotlib/matplotlib", "Colorbar ticks are incorrectly positioned when using LogNorm with vmin=0, causing a ZeroDivisionError in the tick locator."),
    ]

    tasks = []
    for i in range(n):
        repo, issue = issues[i % len(issues)]
        tasks.append({
            "task_id": f"swe_synth_{i:03d}",
            "messages": [
                {"role": "system", "content": "You are an autonomous software engineering agent with bash access. Fix the GitHub issue."},
                {"role": "user", "content": f"Repository: {repo}\n\nIssue: {issue}"},
            ],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} synthetic SWE tasks to {output_path}")


def download_hotpotqa(output_path: str, max_samples: int = 50):
    """Download HotpotQA and convert to JSONL."""
    try:
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    except Exception as e:
        print(f"Could not download HotpotQA: {e}")
        _gen_synthetic_research_questions(output_path, max_samples)
        return

    tasks = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        system_prompt = (
            "You are a research agent with access to web search and page reading tools. "
            "Answer the following question by searching for information, reading relevant pages, "
            "taking notes, and synthesizing a comprehensive answer with citations. "
            "Perform at least 5 searches and read at least 10 pages before answering."
        )

        tasks.append({
            "task_id": row.get("id", f"hotpot_{i:03d}"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["question"]},
            ],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} HotpotQA tasks to {output_path}")


def _gen_synthetic_research_questions(output_path: str, n: int):
    """Fallback for HotpotQA."""
    questions = [
        "What is the relationship between the founder of SpaceX and the CEO of the company that created ChatGPT?",
        "Which country has a larger GDP per capita: the birthplace of Shakespeare or the birthplace of Mozart?",
        "Compare the tallest buildings in New York City and Dubai. Which was completed more recently?",
        "What do the programming languages Python and Ruby have in common regarding their naming origins?",
        "Which Nobel Prize winner in Physics was born in the same country as the inventor of dynamite?",
    ]
    tasks = []
    for i in range(n):
        tasks.append({
            "task_id": f"research_synth_{i:03d}",
            "messages": [
                {"role": "system", "content": "You are a research agent. Search and read pages to answer thoroughly."},
                {"role": "user", "content": questions[i % len(questions)]},
            ],
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} synthetic research tasks to {output_path}")


def download_bird_sql(output_path: str, max_samples: int = 50):
    """Download BIRD-SQL dev set."""
    try:
        from datasets import load_dataset

        ds = load_dataset("xu3kev/BIRD-SQL-data-dev", split="train")
    except Exception as e:
        print(f"Could not download BIRD-SQL: {e}")
        print("Will use generated NL2SQL prompts instead.")
        return

    tasks = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        question = row.get("question", "")
        db_id = row.get("db_id", "unknown")
        evidence = row.get("evidence", "")

        content = f"Database: {db_id}\n\nQuestion: {question}"
        if evidence:
            content += f"\n\nEvidence/Hint: {evidence}"

        tasks.append({
            "task_id": f"bird_{i:03d}",
            "messages": [
                {"role": "system", "content": "You are a SQL expert. Use the available tools to explore the database schema, understand the data, and write the correct SQL query."},
                {"role": "user", "content": content},
            ],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} BIRD-SQL tasks to {output_path}")


def download_code_contests(output_path: str, max_samples: int = 50):
    """Download CodeContests and convert to JSONL."""
    try:
        from datasets import load_dataset

        ds = load_dataset("deepmind/code_contests", split="test")
    except Exception as e:
        print(f"Could not download CodeContests: {e}")
        _gen_synthetic_code_tasks(output_path, max_samples)
        return

    tasks = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        system_prompt = (
            "You are a competitive programming agent with access to a Python code interpreter. "
            "Read the problem, think step by step, write a solution, test it against the examples, "
            "and iterate until all examples pass. Then provide your final solution."
        )

        tasks.append({
            "task_id": f"codecontest_{i:03d}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["description"]},
            ],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} CodeContests tasks to {output_path}")


def _gen_synthetic_code_tasks(output_path: str, n: int):
    """Fallback for CodeContests."""
    problems = [
        "Given an array of integers, find the longest subsequence such that the absolute difference between any two adjacent elements is at most k. Return the length of this subsequence.",
        "You are given a weighted directed graph with n nodes. Find the shortest path from node 1 to node n that visits at least one node from a given set of required nodes.",
        "Implement a data structure that supports: insert(x), delete(x), getRandom() - returns a random element, and getMedian() - returns the median. All operations should be O(log n).",
    ]
    tasks = []
    for i in range(n):
        tasks.append({
            "task_id": f"code_synth_{i:03d}",
            "messages": [
                {"role": "system", "content": "Solve the problem using Python. Test your solution."},
                {"role": "user", "content": problems[i % len(problems)]},
            ],
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} synthetic code tasks to {output_path}")


def download_numina_math(output_path: str, max_samples: int = 50):
    """Download NuminaMath-TIR and convert to JSONL."""
    try:
        from datasets import load_dataset

        ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    except Exception as e:
        print(f"Could not download NuminaMath-TIR: {e}")
        _gen_synthetic_math_tasks(output_path, max_samples)
        return

    tasks = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        system_prompt = (
            "You are a math problem solver with access to a Python code interpreter. "
            "Think through the problem step by step. Use Python to verify your calculations, "
            "explore special cases, and check your answer. Show your work."
        )

        tasks.append({
            "task_id": f"math_{i:03d}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row.get("problem", row.get("question", str(row)))},
            ],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} NuminaMath tasks to {output_path}")


def _gen_synthetic_math_tasks(output_path: str, n: int):
    """Fallback for NuminaMath."""
    problems = [
        "Find all positive integers n such that n^2 + 2n + 12 is a perfect square.",
        "In triangle ABC, angle A = 60 degrees, b = 5, c = 8. Find the exact value of side a.",
        "Evaluate the sum: 1/1*2 + 1/2*3 + 1/3*4 + ... + 1/99*100.",
    ]
    tasks = []
    for i in range(n):
        tasks.append({
            "task_id": f"math_synth_{i:03d}",
            "messages": [
                {"role": "system", "content": "Solve the math problem. Use Python to verify."},
                {"role": "user", "content": problems[i % len(problems)]},
            ],
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} synthetic math tasks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets")
    parser.add_argument("--output-dir", default="benchmark/data")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    d = args.output_dir
    n = args.max_samples

    download_swe_bench(f"{d}/swe_bench.jsonl", n)
    download_hotpotqa(f"{d}/hotpotqa.jsonl", n)
    download_bird_sql(f"{d}/bird_sql.jsonl", n)
    download_code_contests(f"{d}/code_contests.jsonl", n)
    download_numina_math(f"{d}/numina_math.jsonl", n)
