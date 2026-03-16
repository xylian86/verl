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
Mock search executor for deep research tasks.
Returns synthetic but realistic-looking search results and page content.
"""

import hashlib
import random


# Realistic-looking paragraphs to use as page content
_PARAGRAPHS = [
    "Recent advances in large language models have demonstrated remarkable capabilities across a wide range of tasks. These models, trained on vast corpora of text data, exhibit emergent abilities that scale with model size and training compute. The implications for artificial intelligence research are profound, suggesting that continued scaling may yield further breakthroughs in reasoning and understanding.",
    "The database system processes queries through a multi-stage pipeline consisting of parsing, optimization, and execution phases. Query optimization involves both logical transformations (predicate pushdown, join reordering) and physical optimization (index selection, parallel execution planning). Modern optimizers use cost-based approaches that estimate the computational resources required for different execution strategies.",
    "Climate change research indicates that global temperatures have risen by approximately 1.1 degrees Celsius above pre-industrial levels. The Intergovernmental Panel on Climate Change (IPCC) projects that without significant reductions in greenhouse gas emissions, temperatures could rise by 1.5°C as early as 2030. This would have cascading effects on weather patterns, sea levels, and biodiversity.",
    "The company reported quarterly revenue of $45.2 billion, representing a 12% year-over-year increase. Operating margins expanded to 28.5%, driven by improved efficiency in cloud services and enterprise software segments. Management raised full-year guidance, citing strong demand for AI-powered products and services across all geographic regions.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be intractable for classical computers. Current quantum processors contain several hundred qubits, though error rates remain high. Fault-tolerant quantum computing, requiring millions of physical qubits, remains a long-term goal that could revolutionize cryptography, materials science, and drug discovery.",
    "The patient presented with acute onset of chest pain radiating to the left arm, accompanied by diaphoresis and shortness of breath. ECG showed ST-segment elevation in leads V1-V4, consistent with anterior myocardial infarction. The patient was immediately taken to the cardiac catheterization laboratory for percutaneous coronary intervention.",
    "Software engineering best practices emphasize the importance of continuous integration and deployment pipelines. Automated testing at multiple levels—unit, integration, and end-to-end—helps catch regressions early. Code review processes, combined with static analysis tools, further improve code quality and maintainability across large engineering organizations.",
    "The archaeological excavation at the site revealed multiple occupation layers spanning from the Neolithic period through the Bronze Age. Radiocarbon dating of organic remains places the earliest settlement at approximately 4500 BCE. Ceramic typology and lithic analysis suggest connections to contemporary cultures in the wider region.",
]


def _generate_content(query: str, length: int = 3) -> str:
    """Generate deterministic but varied content based on query."""
    seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    selected = rng.sample(_PARAGRAPHS * 3, min(length, len(_PARAGRAPHS) * 3))
    return "\n\n".join(selected)


class MockSearchExecutor:
    def __init__(self, max_results: int = 5, page_length: int = 5, **kwargs):
        self.max_results = max_results
        self.page_length = page_length

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information. Returns a list of search results with titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "Read the full content of a web page given its URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL of the page to read."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "note_take",
                    "description": "Save a note for later reference during research.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title of the note."},
                            "content": {"type": "string", "description": "Content of the note."},
                        },
                        "required": ["title", "content"],
                    },
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "web_search":
            return self._search(args.get("query", ""))
        elif tool_name == "read_page":
            return self._read_page(args.get("url", ""))
        elif tool_name == "note_take":
            return f"Note saved: {args.get('title', 'Untitled')}"
        return f"Unknown tool: {tool_name}"

    def _search(self, query: str) -> str:
        seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        domains = [
            "wikipedia.org",
            "arxiv.org",
            "nature.com",
            "sciencedirect.com",
            "pubmed.ncbi.nlm.nih.gov",
            "reuters.com",
            "bloomberg.com",
            "github.com",
            "stackoverflow.com",
            "medium.com",
        ]

        results = []
        for i in range(self.max_results):
            domain = rng.choice(domains)
            slug = query.lower().replace(" ", "-")[:30]
            snippet = _generate_content(f"{query}_{i}", length=1)[:200]
            results.append(
                {
                    "title": f"{query.title()} - Result {i + 1}",
                    "url": f"https://{domain}/article/{slug}-{i}",
                    "snippet": snippet,
                }
            )

        import json

        return json.dumps(results, indent=2)

    def _read_page(self, url: str) -> str:
        content = _generate_content(url, length=self.page_length)
        return f"# Page Content\nURL: {url}\n\n{content}"
