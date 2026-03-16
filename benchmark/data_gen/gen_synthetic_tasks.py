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
Generate synthetic prompts for use cases that don't have readily available datasets.
Covers: GUI agent, multi-file code generation, deep research, and SWE-agent tasks.
"""

import argparse
import json
import os
import random


def gen_gui_tasks(output_path: str, n: int = 50, seed: int = 42):
    """Generate GUI/computer-use agent tasks."""
    rng = random.Random(seed)

    apps = [
        ("spreadsheet", [
            "Open the monthly sales spreadsheet, filter rows where revenue > $50k, sort by date descending, and export the result as CSV.",
            "Create a pivot table showing total sales by region and product category. Add a chart visualizing the top 5 regions.",
            "Find all cells with formula errors in the Q3 budget sheet. Fix them and recalculate totals.",
            "Copy the 'Summary' tab from last month's report, paste it into this month's file, and update all date references.",
            "Merge the customer data from three different sheets, remove duplicates based on email, and create a consolidated view.",
        ]),
        ("web_app", [
            "Go to the admin dashboard, find users who haven't logged in for 30 days, export their list, then navigate to the email template editor and draft a re-engagement email.",
            "Navigate to the settings page, enable two-factor authentication, configure the backup email, and verify the setup by going through the test flow.",
            "Open the project management tool, create a new sprint, add 5 tasks from the backlog, assign them to team members, and set the sprint dates.",
            "Go to the analytics dashboard, change the date range to last quarter, export the conversion funnel report as PDF, then switch to the revenue tab and export that too.",
            "Navigate to the CMS, find all draft blog posts older than 30 days, move them to archive, and publish the 3 most recent approved posts.",
        ]),
        ("desktop", [
            "Open the file manager, navigate to the project directory, find all .log files larger than 100MB, compress them into a single archive, and move the archive to the backup folder.",
            "Open the IDE, find all TODO comments across the project, create a new file listing them with their file locations, and open a terminal to run the test suite.",
            "Configure the system firewall: add a rule to allow port 8080, block all traffic from subnet 10.0.0.0/8 except the gateway, and export the rules.",
            "Open the email client, search for all emails from 'finance@company.com' in the last month, create a new folder called 'Q4 Reports', and move matching emails there.",
            "Use the image editor to batch-resize all PNG files in the 'uploads' folder to 800x600, convert them to WebP format, and save to a new 'optimized' folder.",
        ]),
    ]

    tasks = []
    i = 0
    while len(tasks) < n:
        app, task_list = rng.choice(apps)
        task_text = rng.choice(task_list)
        # Add complexity variations
        if rng.random() > 0.5:
            task_text += " After completing this, verify the changes were saved correctly."
        if rng.random() > 0.7:
            task_text += " If you encounter any errors, take a screenshot and try an alternative approach."

        tasks.append({
            "task_id": f"gui_{app}_{i:03d}",
            "messages": [{"role": "user", "content": task_text}],
        })
        i += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} GUI tasks to {output_path}")


def gen_deep_research_tasks(output_path: str, n: int = 50, seed: int = 42):
    """Generate deep research tasks requiring multiple search+read cycles."""
    rng = random.Random(seed)

    topics = [
        "Comprehensive analysis of transformer architecture innovations from 2023-2024: compare Mamba, RWKV, RetNet, and standard transformers on efficiency, quality, and scaling properties.",
        "Research the current state of nuclear fusion energy: compare ITER, NIF, and private companies (Commonwealth Fusion, TAE Technologies). Include timeline projections and technical challenges.",
        "Investigate the global semiconductor supply chain: identify bottlenecks, geopolitical risks, and the impact of CHIPS Act and EU Chips Act on reshoring efforts.",
        "Analyze the rise of sovereign AI: compare national AI strategies of US, China, EU, UK, India, and UAE. Include investment levels, regulatory approaches, and key initiatives.",
        "Research the current state of CRISPR gene therapy: catalog all FDA-approved treatments, those in Phase 3 trials, and the main technical and ethical challenges remaining.",
        "Compare the top 5 open-source LLMs (Llama 3, Mistral, Qwen, DeepSeek, Gemma) across benchmarks, architecture choices, training data, and commercial licensing.",
        "Investigate the economics of data centers for AI: energy consumption trends, cooling technologies, geographic distribution, and projected growth through 2030.",
        "Research quantum error correction approaches: compare surface codes, color codes, and topological approaches. Include recent experimental milestones from IBM, Google, and Microsoft.",
        "Analyze the impact of AI on drug discovery: catalog major AI-discovered drugs in clinical trials, success rates vs traditional methods, and leading companies/platforms.",
        "Research the global lithium market: supply chain from mining to battery production, price trends, recycling technologies, and alternatives (sodium-ion, solid-state).",
        "Investigate autonomous vehicle regulation worldwide: compare US state-by-state rules, EU regulations, China's framework, and incidents/safety data.",
        "Research the state of protein structure prediction post-AlphaFold: RoseTTAFold, ESMFold, competing approaches, impact on drug discovery and biology.",
        "Analyze the carbon credit market: compare voluntary vs compliance markets, major exchanges, pricing trends, concerns about quality, and blockchain-based solutions.",
        "Research synthetic biology applications: biomanufacturing, sustainable materials, food production, and biosecurity concerns.",
        "Investigate the state of brain-computer interfaces: compare Neuralink, Synchron, Blackrock, and academic approaches. Include clinical results and ethical considerations.",
    ]

    tasks = []
    for i in range(n):
        topic = topics[i % len(topics)]
        tasks.append({
            "task_id": f"research_{i:03d}",
            "messages": [{"role": "user", "content": f"Please conduct a thorough research report on the following topic. Search for the latest information, read multiple sources, cross-reference claims, and produce a well-structured report with citations.\n\nTopic: {topic}"}],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} research tasks to {output_path}")


def gen_multi_file_codegen_tasks(output_path: str, n: int = 50, seed: int = 42):
    """Generate multi-file code generation tasks."""
    rng = random.Random(seed)

    tasks_data = [
        "Build a REST API for a task management system with the following requirements:\n- User authentication (JWT tokens)\n- CRUD operations for projects and tasks\n- Task assignment and status tracking\n- Due date reminders\n- Write unit tests for all endpoints\n- Use Python with FastAPI\n- Include a Dockerfile and docker-compose.yml",

        "Create a CLI tool for analyzing Git repositories:\n- Parse git log to extract commit statistics\n- Show contributor activity over time\n- Detect large files and suggest .gitignore entries\n- Generate a markdown report\n- Include comprehensive tests\n- Use Python with Click",

        "Build a real-time chat application:\n- WebSocket-based messaging\n- Multiple chat rooms\n- Message persistence with SQLite\n- User authentication\n- Rate limiting\n- Write integration tests\n- Use Python with websockets library",

        "Create a data pipeline framework:\n- Define pipeline steps as decorators\n- Support parallel execution of independent steps\n- Error handling and retry logic\n- Progress tracking and logging\n- Configuration via YAML files\n- Comprehensive test suite\n- Use Python with asyncio",

        "Build a markdown-to-HTML static site generator:\n- Parse markdown files with frontmatter\n- Jinja2 template support\n- Asset pipeline (CSS, JS minification)\n- RSS feed generation\n- Sitemap generation\n- Watch mode for development\n- Full test coverage",

        "Create a database migration tool:\n- Parse SQL migration files\n- Track applied migrations\n- Support up/down migrations\n- Dry-run mode\n- Multiple database backend support (SQLite, PostgreSQL)\n- Conflict detection\n- Write tests",

        "Build a load testing tool:\n- HTTP request definition via YAML\n- Concurrent request execution\n- Response time percentiles (p50, p95, p99)\n- Error rate tracking\n- HTML report generation\n- Rate limiting support\n- Tests for core logic",

        "Create a Python package dependency analyzer:\n- Parse requirements.txt and pyproject.toml\n- Build dependency graph\n- Detect version conflicts\n- Check for known vulnerabilities (via API)\n- Generate visualization\n- Export report in multiple formats\n- Unit tests",

        "Build a log analysis tool:\n- Parse multiple log formats (JSON, syslog, Apache)\n- Pattern detection and anomaly highlighting\n- Time-range filtering\n- Aggregation and statistics\n- Export to CSV/JSON\n- Interactive CLI mode\n- Comprehensive tests",

        "Create a configuration management tool:\n- YAML/JSON/TOML config file support\n- Environment variable interpolation\n- Config validation via schemas\n- Secrets masking in logs\n- Config diff between environments\n- Inheritance/overlay support\n- Full test suite",
    ]

    tasks = []
    for i in range(n):
        task = tasks_data[i % len(tasks_data)]
        # Add variation
        if rng.random() > 0.5:
            task += "\n\nPlease also add type hints and docstrings throughout."
        if rng.random() > 0.7:
            task += "\n\nInclude a Makefile with common development commands."

        tasks.append({
            "task_id": f"codegen_{i:03d}",
            "messages": [{"role": "user", "content": task}],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} code generation tasks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui-output", default="benchmark/data/gui_tasks.jsonl")
    parser.add_argument("--research-output", default="benchmark/data/research_tasks.jsonl")
    parser.add_argument("--codegen-output", default="benchmark/data/codegen_tasks.jsonl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen_gui_tasks(args.gui_output, n=args.n, seed=args.seed)
    gen_deep_research_tasks(args.research_output, n=args.n, seed=args.seed)
    gen_multi_file_codegen_tasks(args.codegen_output, n=args.n, seed=args.seed)
