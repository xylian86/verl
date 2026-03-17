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
Real search executor for deep research tasks.
Uses DuckDuckGo for web search and requests+BeautifulSoup for page reading.
No API keys required.

Install dependencies:
    pip install duckduckgo-search beautifulsoup4 requests
"""

import json
import re
import traceback

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


class RealSearchExecutor:
    def __init__(self, max_results: int = 5, max_page_chars: int = 8000, timeout: int = 10, **kwargs):
        self.max_results = max_results
        self.max_page_chars = max_page_chars
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        })
        self.notes: list[dict] = []

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
                    "description": "Read the full content of a web page given its URL. Returns the main text content.",
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
            title = args.get("title", "Untitled")
            content = args.get("content", "")
            self.notes.append({"title": title, "content": content})
            return f"Note saved: {title} ({len(self.notes)} notes total)"
        return f"Unknown tool: {tool_name}"

    def _search(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=self.max_results))

            results = []
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
            return json.dumps(results, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)[:200]}"})

    def _read_page(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"# Page Content\nURL: {url}\n\n[Non-HTML content: {content_type}]"

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove non-content elements
            for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
                tag.decompose()

            # Try to find main content area
            main = soup.find("main") or soup.find("article") or soup.find(role="main")
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                body = soup.find("body")
                text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

            # Clean up whitespace: collapse blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            if len(text) > self.max_page_chars:
                text = text[: self.max_page_chars] + "\n\n[... truncated ...]"

            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            header = f"# {title}\nURL: {url}\n\n" if title else f"# Page Content\nURL: {url}\n\n"
            return header + text

        except requests.exceptions.Timeout:
            return f"# Page Content\nURL: {url}\n\n[Error: Request timed out after {self.timeout}s]"
        except requests.exceptions.RequestException as e:
            return f"# Page Content\nURL: {url}\n\n[Error: {str(e)[:200]}]"
        except Exception:
            return f"# Page Content\nURL: {url}\n\n[Error: {traceback.format_exc()[:200]}]"
