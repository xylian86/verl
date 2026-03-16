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
GUI simulator executor for computer-use agent tasks.
Returns synthetic accessibility-tree-like observations.
"""

import hashlib
import json
import random


_ELEMENTS = [
    {"tag": "button", "text": "Submit", "id": "btn-submit", "bbox": [100, 200, 180, 40]},
    {"tag": "input", "text": "", "id": "search-box", "type": "text", "bbox": [50, 50, 300, 30]},
    {"tag": "a", "text": "Home", "id": "nav-home", "href": "/", "bbox": [10, 10, 60, 20]},
    {"tag": "a", "text": "Settings", "id": "nav-settings", "href": "/settings", "bbox": [80, 10, 80, 20]},
    {"tag": "div", "text": "Welcome to the application dashboard.", "id": "main-content", "bbox": [50, 100, 500, 300]},
    {"tag": "select", "text": "Choose option", "id": "dropdown-1", "options": ["Option A", "Option B", "Option C"], "bbox": [50, 300, 200, 30]},
    {"tag": "table", "text": "Data Table", "id": "data-table", "rows": 25, "cols": 6, "bbox": [50, 350, 700, 400]},
    {"tag": "textarea", "text": "", "id": "editor", "bbox": [50, 150, 500, 200]},
    {"tag": "button", "text": "Cancel", "id": "btn-cancel", "bbox": [200, 200, 100, 40]},
    {"tag": "div", "text": "Loading...", "id": "loader", "bbox": [250, 250, 100, 30]},
    {"tag": "img", "alt": "Chart visualization", "id": "chart-1", "bbox": [50, 500, 400, 300]},
    {"tag": "button", "text": "Export CSV", "id": "btn-export", "bbox": [500, 50, 120, 35]},
    {"tag": "input", "text": "", "id": "date-picker", "type": "date", "bbox": [350, 50, 150, 30]},
    {"tag": "div", "text": "Error: Connection timeout. Please retry.", "id": "error-banner", "bbox": [0, 0, 800, 40]},
    {"tag": "button", "text": "Retry", "id": "btn-retry", "bbox": [700, 5, 80, 30]},
]


def _generate_observation(state_hash: str, n_elements: int = 10) -> str:
    """Generate a synthetic accessibility tree observation."""
    seed = int(hashlib.md5(state_hash.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    selected = rng.sample(_ELEMENTS, min(n_elements, len(_ELEMENTS)))

    lines = ["=== Accessibility Tree ===", f"URL: https://app.example.com/page/{state_hash[:8]}", ""]
    for i, elem in enumerate(selected):
        attrs = " ".join(f'{k}="{v}"' for k, v in elem.items() if k not in ("tag", "text", "bbox"))
        lines.append(f"[{i}] <{elem['tag']} {attrs}> {elem.get('text', '')}")
        lines.append(f"    bbox: {elem['bbox']}")
    lines.append(f"\nViewport: 1280x800, Scroll position: (0, {rng.randint(0, 500)})")
    return "\n".join(lines)


class GUISimulatorExecutor:
    def __init__(self, **kwargs):
        self._state = "initial"
        self._action_count = 0

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click on an element by its ID or coordinates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "ID of the element to click."},
                            "x": {"type": "integer", "description": "X coordinate (optional if element_id provided)."},
                            "y": {"type": "integer", "description": "Y coordinate (optional if element_id provided)."},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text into the currently focused element or specified element.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to type."},
                            "element_id": {"type": "string", "description": "Element ID to focus before typing."},
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scroll",
                    "description": "Scroll the page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction."},
                            "amount": {"type": "integer", "description": "Pixels to scroll."},
                        },
                        "required": ["direction"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "screenshot",
                    "description": "Take a screenshot of the current page. Returns accessibility tree representation.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "navigate",
                    "description": "Navigate to a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to navigate to."},
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        self._action_count += 1
        self._state = f"{tool_name}_{self._action_count}_{json.dumps(args, sort_keys=True)[:50]}"

        if tool_name == "click":
            elem = args.get("element_id", f"({args.get('x', 0)},{args.get('y', 0)})")
            obs = _generate_observation(self._state)
            return f"Clicked on {elem}. Page updated.\n\n{obs}"

        elif tool_name == "type_text":
            obs = _generate_observation(self._state)
            return f"Typed '{args.get('text', '')[:50]}' into {args.get('element_id', 'focused element')}.\n\n{obs}"

        elif tool_name == "scroll":
            obs = _generate_observation(self._state)
            return f"Scrolled {args.get('direction', 'down')} {args.get('amount', 300)}px.\n\n{obs}"

        elif tool_name == "screenshot":
            return _generate_observation(self._state, n_elements=12)

        elif tool_name == "navigate":
            obs = _generate_observation(self._state)
            return f"Navigated to {args.get('url', '')}. Page loaded.\n\n{obs}"

        return f"Unknown tool: {tool_name}"
