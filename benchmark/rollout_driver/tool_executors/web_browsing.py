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
Web browsing executor for AgentGym-style web tasks.
Combines mock search with page navigation and form interaction.
"""

import hashlib
import json
import random


def _gen_page(url: str, length: int = 800) -> str:
    """Generate deterministic fake page content."""
    seed = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    titles = [
        "Product Details", "Search Results", "User Profile",
        "Shopping Cart", "Order History", "Help Center",
        "Documentation", "API Reference", "Dashboard",
    ]
    sections = [
        "Overview", "Details", "Reviews", "Related Items",
        "Specifications", "Pricing", "FAQ", "Contact",
    ]

    title = rng.choice(titles)
    lines = [f"# {title}", f"URL: {url}", ""]

    for _ in range(rng.randint(3, 6)):
        section = rng.choice(sections)
        lines.append(f"## {section}")
        # Generate paragraph-like content
        words = ["the", "data", "system", "provides", "analysis", "results",
                 "performance", "metrics", "indicate", "significant", "improvement",
                 "compared", "to", "baseline", "approach", "using", "advanced",
                 "techniques", "for", "optimization", "and", "processing"]
        for _ in range(rng.randint(2, 4)):
            sentence = " ".join(rng.choices(words, k=rng.randint(10, 20)))
            lines.append(sentence.capitalize() + ".")
        lines.append("")

    # Add some interactive elements
    lines.append("### Interactive Elements:")
    elements = [
        "[button: Search] [input: search-query] [button: Filter]",
        "[link: Next Page] [link: Previous Page] [select: Sort By]",
        "[button: Add to Cart] [button: Buy Now] [link: More Details]",
    ]
    lines.append(rng.choice(elements))

    return "\n".join(lines)


class WebBrowsingExecutor:
    def __init__(self, **kwargs):
        self._current_url = "https://example.com"
        self._history = []

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "goto",
                    "description": "Navigate to a URL and return the page content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to navigate to."},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web and return results.",
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
                    "name": "click_link",
                    "description": "Click a link on the current page by its text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "link_text": {"type": "string", "description": "Text of the link to click."},
                        },
                        "required": ["link_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fill_form",
                    "description": "Fill in a form field on the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string", "description": "Name of the form field."},
                            "value": {"type": "string", "description": "Value to fill in."},
                        },
                        "required": ["field_name", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_form",
                    "description": "Submit the current form.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "go_back",
                    "description": "Go back to the previous page.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "goto":
            url = args.get("url", "https://example.com")
            self._history.append(self._current_url)
            self._current_url = url
            return _gen_page(url)

        elif tool_name == "search":
            query = args.get("query", "")
            seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            results = []
            for i in range(5):
                slug = query.lower().replace(" ", "-")[:20]
                results.append({
                    "title": f"{query.title()} - Result {i+1}",
                    "url": f"https://example.com/{slug}/{rng.randint(1000, 9999)}",
                    "snippet": f"Information about {query} including detailed analysis and results...",
                })
            return json.dumps(results, indent=2)

        elif tool_name == "click_link":
            link = args.get("link_text", "")
            new_url = f"{self._current_url}/{link.lower().replace(' ', '-')}"
            self._history.append(self._current_url)
            self._current_url = new_url
            return _gen_page(new_url)

        elif tool_name == "fill_form":
            return f"Filled '{args.get('field_name', '')}' with '{args.get('value', '')[:50]}'"

        elif tool_name == "submit_form":
            new_url = self._current_url + "/submitted"
            self._history.append(self._current_url)
            self._current_url = new_url
            return f"Form submitted.\n\n{_gen_page(new_url)}"

        elif tool_name == "go_back":
            if self._history:
                self._current_url = self._history.pop()
                return _gen_page(self._current_url)
            return "No previous page."

        return f"Unknown tool: {tool_name}"
