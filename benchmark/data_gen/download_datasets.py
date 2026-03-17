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


def download_research_data(output_path: str, max_samples: int = 50):
    """Download real research tasks from FRAMES (multi-hop retrieval) and ELI5 (long-form research).

    FRAMES provides 824 challenging multi-hop questions requiring 2-15 Wikipedia articles.
    ELI5 provides open-ended "Explain Like I'm 5" questions that demand thorough research.
    We mix both to get diverse research-style prompts.
    """
    tasks = []

    # --- FRAMES: multi-hop retrieval questions ---
    frames_count = 0
    try:
        from datasets import load_dataset

        ds = load_dataset("google/frames-benchmark", split="test")
        system_prompt = (
            "You are a deep research agent with access to web search and page reading tools. "
            "Given a research question, you must: (1) search for information from multiple angles, "
            "(2) read at least 10 web pages, (3) take notes on key findings, (4) cross-reference "
            "claims across sources, and (5) produce a comprehensive report with citations. "
            "Be thorough — explore follow-up questions that arise during research."
        )

        for i, row in enumerate(ds):
            if len(tasks) >= max_samples:
                break
            prompt = row.get("Prompt") or row.get("prompt") or row.get("question", "")
            if not prompt:
                continue
            tasks.append({
                "task_id": f"frames_{i:03d}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            })
            frames_count += 1
    except Exception as e:
        print(f"Could not download FRAMES: {e}")

    # --- ELI5: long-form research questions (fill remaining slots) ---
    eli5_count = 0
    remaining = max_samples - len(tasks)
    if remaining > 0:
        try:
            from datasets import load_dataset

            ds = load_dataset("rexarski/eli5_category", split="validation")
            system_prompt = (
                "You are a deep research agent with access to web search and page reading tools. "
                "Given a question, conduct thorough research: search from multiple angles, "
                "read at least 10 web pages, take notes, cross-reference claims, and produce "
                "a comprehensive, well-structured answer with citations."
            )

            # Filter to science/technology categories for more research-worthy questions
            research_categories = {
                "Biology", "Chemistry", "Earth Science", "Economics",
                "Engineering", "Mathematics", "Physics", "Technology",
            }

            for i, row in enumerate(ds):
                if eli5_count >= remaining:
                    break
                category = row.get("category", "")
                if category not in research_categories:
                    continue
                title = row.get("title", "")
                selftext = row.get("selftext", "")
                question = title
                if selftext and selftext.strip():
                    question += f"\n\n{selftext.strip()}"
                if not question.strip():
                    continue
                tasks.append({
                    "task_id": f"eli5_{eli5_count:03d}",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                })
                eli5_count += 1
        except Exception as e:
            print(f"Could not download ELI5: {e}")

    if not tasks:
        print("WARNING: Could not download any real research data. Falling back to synthetic.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} research tasks to {output_path} (FRAMES: {frames_count}, ELI5: {eli5_count})")


def download_agentgym_rl(output_path: str, max_samples: int = 50):
    """Download AgentGym-RL tasks across all 5 environments.

    Tries the HuggingFace dataset first; falls back to synthetic generation
    covering WebArena, SearchQA, TextCraft, BabyAI, and SciWorld.
    """
    tasks = []

    try:
        from datasets import load_dataset

        ds = load_dataset("AgentGym/AgentTraj-L", split="train")
        env_map = {
            "webarena": "webarena",
            "searchqa": "searchqa",
            "textcraft": "textcraft",
            "babyai": "babyai",
            "sciworld": "sciworld",
        }
        per_env = max(1, max_samples // 5)
        env_counts: dict[str, int] = {e: 0 for e in env_map}

        for row in ds:
            env_raw = (row.get("environment") or row.get("env_name") or "").lower()
            matched_env = None
            for key in env_map:
                if key in env_raw:
                    matched_env = key
                    break
            if matched_env is None:
                continue
            if env_counts[matched_env] >= per_env:
                continue

            conversation = row.get("conversations") or row.get("messages") or []
            if isinstance(conversation, str):
                conversation = [{"role": "user", "content": conversation}]
            if not conversation:
                continue

            tasks.append({
                "task_id": f"agentgym_{matched_env}_{env_counts[matched_env]:03d}",
                "messages": conversation,
                "environment": matched_env,
            })
            env_counts[matched_env] += 1

            if len(tasks) >= max_samples:
                break

        if tasks:
            print(f"Downloaded {len(tasks)} AgentGym-RL tasks: {dict(env_counts)}")
    except Exception as e:
        print(f"Could not download AgentGym-RL dataset: {e}")

    if not tasks:
        print("Generating synthetic AgentGym-RL tasks for all 5 environments.")
        tasks = _gen_synthetic_agentgym(max_samples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(tasks)} AgentGym-RL tasks to {output_path}")


def _gen_synthetic_agentgym(n: int) -> list[dict]:
    """Generate synthetic tasks across all 5 AgentGym-RL environments."""
    env_configs = {
        "webarena": {
            "system": (
                "You are a web navigation agent in a WebArena environment. "
                "Navigate web pages, click elements, fill forms, and search to complete the task. "
                "Observe the accessibility tree after each action to understand the page state."
            ),
            "tasks": [
                "Find the cheapest laptop with at least 16GB RAM on the shopping site and add it to cart.",
                "Navigate to the user forums and find the most recent post about shipping delays.",
                "Search for 'wireless headphones', filter by price under $50, and compare the top 3 results.",
                "Log in with the test account, go to order history, and find the tracking number for order #1234.",
                "Find the return policy page and summarize the conditions for electronics returns.",
            ],
        },
        "searchqa": {
            "system": (
                "You are a SearchQA agent. Use search and document reading tools to find the answer "
                "to the given question. Search from multiple angles and verify across sources."
            ),
            "tasks": [
                "What year was the first successful organ transplant performed, and who was the surgeon?",
                "How many UNESCO World Heritage Sites are there in Italy as of 2024?",
                "What is the chemical formula of the compound commonly known as baking soda, and what is its primary industrial use?",
                "Which country hosted the first Winter Olympics, and in what year?",
                "What is the tallest waterfall in the world and where is it located?",
            ],
        },
        "textcraft": {
            "system": (
                "You are an agent in a TextCraft environment, a text-based crafting game. "
                "Gather materials, learn recipes, and craft the requested item. "
                "Check your inventory and available recipes to plan your crafting path."
            ),
            "tasks": [
                "Craft a stone sword. You may need to gather materials first.",
                "Craft a bookshelf. Plan the full material chain needed.",
                "Create 5 torches using available resources.",
                "Craft a bow and 10 arrows for hunting.",
                "Build a furnace and use it to smelt iron ore into iron ingots.",
            ],
        },
        "babyai": {
            "system": (
                "You are an agent in a BabyAI grid world. Navigate the grid, pick up objects, "
                "and interact with doors/boxes to complete the given instruction. "
                "Observe your surroundings and plan an efficient path."
            ),
            "tasks": [
                "Go to the red ball.",
                "Pick up the blue key and open the blue door.",
                "Put the green ball next to the purple box.",
                "Pick up the yellow key, then go to the grey ball.",
                "Open the red door, then pick up the blue ball behind it.",
            ],
        },
        "sciworld": {
            "system": (
                "You are a science experiment agent in SciWorld. Conduct experiments by "
                "moving between labs, collecting equipment and materials, and performing "
                "procedures to answer the scientific question. Record your findings."
            ),
            "tasks": [
                "Determine whether salt or sugar dissolves faster in water at room temperature.",
                "Measure the pH of three different soil samples and rank them from most acidic to most basic.",
                "Heat water to boiling and record the temperature at which it begins to boil.",
                "Mix an acid and a base together and observe the reaction. Use an indicator to verify neutralization.",
                "Grow a plant seed with and without fertilizer and compare growth after 5 simulated days.",
            ],
        },
    }

    tasks = []
    per_env = max(1, n // 5)
    for env_name, cfg in env_configs.items():
        for i in range(per_env):
            task_text = cfg["tasks"][i % len(cfg["tasks"])]
            tasks.append({
                "task_id": f"agentgym_{env_name}_{i:03d}",
                "environment": env_name,
                "messages": [
                    {"role": "system", "content": cfg["system"]},
                    {"role": "user", "content": task_text},
                ],
            })
    return tasks[:n]


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
    download_research_data(f"{d}/research_tasks_real.jsonl", n)
    download_agentgym_rl(f"{d}/agentgym_rl_tasks.jsonl", n)
