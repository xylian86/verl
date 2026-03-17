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
AgentGym-RL multi-environment executor.

Simulates the five AgentGym-RL environments for benchmarking sequence lengths:
  - WebArena: web navigation with HTML observations
  - SearchQA: search and retrieval question answering
  - TextCraft: text-based crafting/survival game
  - BabyAI: grid-world navigation
  - SciWorld: science experiment simulation
"""

import hashlib
import json
import random


def _deterministic_rng(seed_str: str) -> random.Random:
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


# ---------------------------------------------------------------------------
# WebArena environment simulation
# ---------------------------------------------------------------------------

_HTML_TEMPLATES = [
    '<div class="product-list">{items}</div><nav>{links}</nav>',
    '<main class="content"><h1>{title}</h1><article>{body}</article></main><aside>{sidebar}</aside>',
    '<div class="search-results">{items}</div><div class="pagination">{links}</div>',
    '<form id="checkout">{fields}</form><div class="summary">{body}</div>',
]


def _gen_webarena_obs(url: str, action: str = "", min_elements: int = 40, min_chars: int = 3000) -> str:
    """Generate a realistically-sized accessibility tree observation.

    Real WebArena pages typically produce 2K-10K tokens of accessibility tree.
    """
    rng = _deterministic_rng(url + action)
    tag_types = ["button", "link", "input", "select", "textarea", "div", "span", "a",
                 "img", "heading", "listitem", "tab", "checkbox", "radio", "menuitem"]
    label_words = [
        "Shop", "Cart", "Login", "Search", "Filter", "Sort", "Next", "Details",
        "Add", "Remove", "Price", "Rating", "Review", "Category", "Home", "Wishlist",
        "Compare", "Checkout", "Continue", "Back", "Apply", "Clear", "Save",
        "Delete", "Edit", "View", "Download", "Upload", "Submit", "Cancel",
        "Shipping", "Payment", "Address", "Account", "Orders", "Settings",
        "Help", "FAQ", "Contact", "About", "Privacy", "Terms", "Return",
        "Refund", "Track", "Notify", "Subscribe", "Unsubscribe", "Share",
    ]

    n_elements = rng.randint(max(40, min_elements), min_elements + 60)
    items = []
    for i in range(n_elements):
        tag = rng.choice(tag_types)
        n_words = rng.randint(1, 8)
        text = " ".join(rng.choices(label_words, k=n_words))
        attrs = []
        if tag == "input":
            attrs.append(f'type="{rng.choice(["text", "email", "password", "number", "tel"])}"')
            attrs.append(f'placeholder="{" ".join(rng.choices(label_words, k=rng.randint(2, 4)))}"')
        elif tag == "img":
            attrs.append(f'alt="{" ".join(rng.choices(label_words, k=rng.randint(3, 8)))}"')
        elif tag == "a":
            slug = "-".join(rng.choices(label_words, k=3)).lower()
            attrs.append(f'href="/{slug}/{rng.randint(100, 9999)}"')
        if rng.random() < 0.3:
            attrs.append(f'class="{rng.choice(["primary", "secondary", "active", "disabled", "highlight"])}"')

        attr_str = " " + " ".join(attrs) if attrs else ""
        items.append(f'[{i}] <{tag} id="el_{i}"{attr_str}> {text} </{tag}>')

    obs = f"URL: {url}\n\nAccessibility Tree:\n" + "\n".join(items)

    # Pad with page text content to reach realistic size
    content_words = [
        "product", "item", "description", "available", "shipping", "free", "stock",
        "delivery", "discount", "offer", "sale", "new", "price", "quantity", "total",
        "subtotal", "tax", "estimated", "business", "days", "standard", "express",
        "overnight", "tracking", "number", "order", "confirmed", "processing",
        "shipped", "delivered", "returned", "refunded", "warranty", "manufacturer",
        "brand", "model", "color", "size", "weight", "dimensions", "material",
        "compatible", "accessories", "recommended", "popular", "trending", "clearance",
        "limited", "exclusive", "member", "rewards", "points", "earned", "redeemed",
    ]
    while len(obs) < min_chars:
        section_title = " ".join(rng.choices(label_words, k=rng.randint(2, 4))).title()
        paragraphs = []
        for _ in range(rng.randint(2, 5)):
            sentence_count = rng.randint(2, 5)
            sentences = []
            for _ in range(sentence_count):
                s = " ".join(rng.choices(content_words, k=rng.randint(8, 20)))
                sentences.append(s.capitalize() + ".")
            paragraphs.append(" ".join(sentences))
        obs += f"\n\n## {section_title}\n" + "\n\n".join(paragraphs)

    return obs


# ---------------------------------------------------------------------------
# SearchQA environment simulation
# ---------------------------------------------------------------------------

def _gen_search_results(query: str, n: int = 10) -> str:
    rng = _deterministic_rng(query)
    results = []
    for i in range(n):
        title_words = rng.choices(
            ["analysis", "study", "research", "findings", "report", "overview",
             "guide", "review", "comparison", "data", "evidence", "survey",
             "comprehensive", "systematic", "empirical", "experimental", "quantitative"],
            k=rng.randint(4, 10),
        )
        snippet_words = rng.choices(
            ["the", "results", "show", "that", "compared", "with", "previous",
             "methods", "our", "approach", "achieves", "significant", "improvement",
             "in", "performance", "across", "multiple", "benchmarks", "and", "datasets",
             "we", "propose", "novel", "framework", "based", "on", "transformer",
             "architecture", "which", "leverages", "attention", "mechanisms",
             "to", "capture", "long-range", "dependencies", "effectively"],
            k=rng.randint(40, 80),
        )
        results.append({
            "rank": i + 1,
            "title": " ".join(title_words).title(),
            "url": f"https://example.com/doc/{rng.randint(1000, 9999)}",
            "snippet": " ".join(snippet_words).capitalize() + ".",
        })
    return json.dumps(results, indent=2)


def _gen_document_content(doc_id: str, min_chars: int = 4000) -> str:
    """Generate realistically-sized document content (typical web article: 1K-3K tokens)."""
    rng = _deterministic_rng(doc_id)
    vocab = [
        "the", "experimental", "results", "demonstrate", "that", "proposed",
        "method", "outperforms", "baseline", "approaches", "by", "a", "margin",
        "of", "approximately", "points", "on", "standard", "evaluation", "metrics",
        "including", "accuracy", "precision", "recall", "F1", "score", "across",
        "diverse", "domains", "and", "task", "configurations", "furthermore",
        "ablation", "studies", "reveal", "each", "component", "contributes",
        "significantly", "to", "overall", "performance", "we", "observe", "consistent",
        "improvements", "when", "scaling", "model", "size", "from", "parameters",
        "training", "data", "augmentation", "strategies", "play", "critical", "role",
        "in", "achieving", "state-of-the-art", "results", "particularly", "for",
        "low-resource", "settings", "where", "labeled", "is", "scarce",
        "our", "analysis", "reveals", "several", "key", "insights", "first",
        "pre-training", "on", "large-scale", "corpora", "provides", "strong",
        "foundation", "second", "fine-tuning", "with", "task-specific", "objectives",
        "yields", "substantial", "gains", "third", "multi-task", "learning",
        "enables", "effective", "knowledge", "transfer", "between", "related", "tasks",
        "however", "challenges", "remain", "distribution", "shift", "adversarial",
        "robustness", "computational", "efficiency", "future", "work", "will",
        "address", "these", "limitations", "through", "novel", "architectures",
    ]
    section_titles = [
        "Abstract", "Introduction", "Related Work", "Methodology",
        "Experimental Setup", "Results and Discussion", "Analysis",
        "Ablation Study", "Conclusion", "References",
    ]

    sections = []
    for title in rng.sample(section_titles, k=rng.randint(4, 8)):
        paragraphs = []
        for _ in range(rng.randint(2, 5)):
            sentences = []
            for _ in range(rng.randint(3, 7)):
                s = " ".join(rng.choices(vocab, k=rng.randint(12, 30)))
                sentences.append(s.capitalize() + ".")
            paragraphs.append(" ".join(sentences))
        sections.append(f"## {title}\n\n" + "\n\n".join(paragraphs))

    content = "\n\n".join(sections)
    while len(content) < min_chars:
        extra_sentences = " ".join(
            (" ".join(rng.choices(vocab, k=rng.randint(12, 25))).capitalize() + ".")
            for _ in range(5)
        )
        content += "\n\n" + extra_sentences
    return content


# ---------------------------------------------------------------------------
# TextCraft environment simulation
# ---------------------------------------------------------------------------

_TEXTCRAFT_ITEMS = [
    "wood_plank", "stick", "stone", "iron_ingot", "gold_ingot", "diamond",
    "coal", "leather", "string", "feather", "flint", "clay_ball",
    "paper", "book", "glass", "redstone", "lapis_lazuli", "emerald",
]

_TEXTCRAFT_RECIPES = {
    "crafting_table": ["wood_plank", "wood_plank", "wood_plank", "wood_plank"],
    "wooden_sword": ["wood_plank", "wood_plank", "stick"],
    "stone_sword": ["stone", "stone", "stick"],
    "iron_sword": ["iron_ingot", "iron_ingot", "stick"],
    "wooden_pickaxe": ["wood_plank", "wood_plank", "wood_plank", "stick", "stick"],
    "furnace": ["stone"] * 8,
    "torch": ["coal", "stick"],
    "bow": ["stick", "stick", "stick", "string", "string", "string"],
    "arrow": ["flint", "stick", "feather"],
    "bookshelf": ["wood_plank"] * 6 + ["book"] * 3,
}


def _gen_textcraft_obs(inventory: dict, location: str) -> str:
    rng = _deterministic_rng(location + str(sorted(inventory.items())))
    inv_str = ", ".join(f"{k}: {v}" for k, v in inventory.items()) if inventory else "(empty)"

    nearby_items = rng.sample(_TEXTCRAFT_ITEMS, k=rng.randint(4, 10))
    nearby_str = ", ".join(f"{item} (x{rng.randint(1,5)})" for item in nearby_items)

    surroundings = rng.choices(
        ["a tall oak tree", "a stone outcrop", "a flowing river", "a dark cave entrance",
         "a grassy clearing", "a sandy beach", "a snow-covered hill", "a mushroom patch",
         "a flower meadow", "a ruined wall", "an abandoned mineshaft", "a lava pool nearby",
         "a dense thicket of bushes", "a shallow pond", "a steep cliff face"],
        k=rng.randint(3, 6),
    )

    craftable = [name for name, mats in _TEXTCRAFT_RECIPES.items()
                 if all(inventory.get(m, 0) >= mats.count(m) for m in set(mats))]
    craftable_str = ", ".join(craftable) if craftable else "none (need more materials)"

    return (
        f"=== TextCraft World ===\n"
        f"Location: {location}\n"
        f"Time: {'day' if rng.random() > 0.3 else 'night'}\n"
        f"Weather: {rng.choice(['clear', 'cloudy', 'raining', 'snowing', 'foggy'])}\n"
        f"Health: {rng.randint(14, 20)}/20  Hunger: {rng.randint(10, 20)}/20\n\n"
        f"Inventory: {inv_str}\n\n"
        f"Nearby items on ground: {nearby_str}\n"
        f"Surroundings: {', '.join(surroundings)}\n"
        f"You see: a crafting table, a furnace, some scattered materials.\n\n"
        f"Currently craftable with inventory: {craftable_str}\n"
        f"Available commands: get, craft, inventory, look, move, recipes"
    )


# ---------------------------------------------------------------------------
# BabyAI environment simulation
# ---------------------------------------------------------------------------

_BABYAI_OBJECTS = ["key", "ball", "box"]
_BABYAI_COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
_BABYAI_DIRS = ["north", "south", "east", "west"]


def _gen_babyai_obs(pos: tuple, grid_size: int = 8) -> str:
    rng = _deterministic_rng(str(pos))

    # Generate a full grid view (7x7 field of vision like real BabyAI)
    visible = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            cell_x, cell_y = pos[0] + dx, pos[1] + dy
            if not (0 <= cell_x < grid_size and 0 <= cell_y < grid_size):
                visible.append(f"  ({cell_x:2d},{cell_y:2d}): wall")
                continue
            if dx == 0 and dy == 0:
                visible.append(f"  ({cell_x:2d},{cell_y:2d}): agent (you)")
                continue
            roll = rng.random()
            if roll < 0.15:
                obj = rng.choice(_BABYAI_OBJECTS)
                color = rng.choice(_BABYAI_COLORS)
                state = ""
                if obj == "box":
                    state = f", {'open' if rng.random() < 0.4 else 'closed'}"
                elif obj == "key":
                    state = ", on floor"
                visible.append(f"  ({cell_x:2d},{cell_y:2d}): {color} {obj}{state}")
            elif roll < 0.25:
                color = rng.choice(_BABYAI_COLORS)
                locked = "locked" if rng.random() < 0.5 else "unlocked"
                visible.append(f"  ({cell_x:2d},{cell_y:2d}): {color} door ({locked})")
            elif roll < 0.30:
                visible.append(f"  ({cell_x:2d},{cell_y:2d}): wall")
            else:
                visible.append(f"  ({cell_x:2d},{cell_y:2d}): empty")

    walls = []
    if pos[0] == 0:
        walls.append("west")
    if pos[0] == grid_size - 1:
        walls.append("east")
    if pos[1] == 0:
        walls.append("south")
    if pos[1] == grid_size - 1:
        walls.append("north")

    facing = rng.choice(["north", "south", "east", "west"])
    obs = (
        f"=== BabyAI Grid World ===\n"
        f"Position: {pos}\n"
        f"Facing: {facing}\n"
        f"Grid size: {grid_size}x{grid_size}\n"
        f"Carrying: nothing\n\n"
        f"Field of vision (7x7):\n" + "\n".join(visible) + "\n\n"
        f"Boundary walls: {', '.join(walls) if walls else 'none'}\n"
        f"Available actions: move_forward, turn_left, turn_right, pick_up, drop, toggle, done"
    )
    return obs


# ---------------------------------------------------------------------------
# SciWorld environment simulation
# ---------------------------------------------------------------------------

_SCIWORLD_ITEMS = [
    "beaker", "thermometer", "bunsen_burner", "test_tube", "graduated_cylinder",
    "petri_dish", "microscope", "pH_meter", "scale", "tongs",
    "water", "salt", "sugar", "acid", "base", "indicator", "soil_sample",
    "plant_seed", "fertilizer", "metal_sample",
]

_SCIWORLD_LOCATIONS = [
    "chemistry_lab", "biology_lab", "physics_lab", "supply_closet",
    "greenhouse", "field_station", "computer_room",
]


def _gen_sciworld_obs(location: str, inventory: list, step: int) -> str:
    rng = _deterministic_rng(location + str(step))
    room_items = rng.sample(_SCIWORLD_ITEMS, k=rng.randint(5, 12))
    inv_str = ", ".join(inventory) if inventory else "(empty)"

    connections = rng.sample(_SCIWORLD_LOCATIONS, k=rng.randint(2, 4))

    item_details = []
    for item in room_items:
        props = []
        if rng.random() < 0.4:
            props.append(f"temperature={rng.randint(15, 95)}°C")
        if rng.random() < 0.3:
            props.append(f"mass={rng.uniform(5.0, 500.0):.1f}g")
        if rng.random() < 0.3:
            props.append(f"volume={rng.uniform(10.0, 1000.0):.1f}mL")
        if rng.random() < 0.2:
            props.append(f"pH={rng.uniform(1.0, 14.0):.1f}")
        if rng.random() < 0.2:
            props.append(f"color={rng.choice(['clear', 'yellow', 'blue', 'red', 'green', 'brown', 'white'])}")
        prop_str = f" ({', '.join(props)})" if props else ""
        item_details.append(f"  - {item}{prop_str}")

    obs = (
        f"=== SciWorld Lab ===\n"
        f"Location: {location}\n"
        f"Step: {step}\n"
        f"Task progress: {'early' if step < 5 else 'mid' if step < 10 else 'late'} stage\n\n"
        f"Inventory: {inv_str}\n\n"
        f"Objects in room:\n" + "\n".join(item_details) + "\n\n"
        f"Exits to: {', '.join(connections)}\n\n"
        f"Lab notebook (recent observations):\n"
    )
    for i in range(rng.randint(2, 5)):
        note = rng.choice([
            f"Step {step - rng.randint(1,3)}: Mixed solution turned {rng.choice(['cloudy', 'clear', 'yellow'])}",
            f"Step {step - rng.randint(1,3)}: Temperature reading = {rng.uniform(20, 100):.1f}°C",
            f"Step {step - rng.randint(1,3)}: pH measured at {rng.uniform(1, 14):.1f}",
            f"Step {step - rng.randint(1,3)}: Mass of sample = {rng.uniform(1, 200):.2f}g",
            f"Step {step - rng.randint(1,3)}: Observed {rng.choice(['precipitate forming', 'gas bubbles', 'color change', 'exothermic reaction', 'crystallization'])}",
        ])
        obs += f"  {note}\n"

    obs += (
        "\nAvailable actions: look, move_to [location], pick_up [item], put_down [item], "
        "pour [item] into [item], heat [item], cool [item], mix [item] with [item], "
        "measure [item], use [item] on [item], read [item], wait, done"
    )
    return obs


# ===========================================================================
# Unified AgentGym executor
# ===========================================================================

class AgentGymExecutor:
    """Multi-environment executor that dispatches to the appropriate environment simulator."""

    VALID_ENVS = {"webarena", "searchqa", "textcraft", "babyai", "sciworld"}

    def __init__(self, environment: str = "webarena", **kwargs):
        if environment not in self.VALID_ENVS:
            raise ValueError(f"Unknown environment '{environment}'. Choose from {self.VALID_ENVS}")
        self.environment = environment
        self._step = 0

        # TextCraft state
        self._tc_inventory: dict[str, int] = {"wood_plank": 4, "stick": 2, "stone": 3}
        self._tc_location = "crafting_area"

        # BabyAI state
        self._bi_pos = (1, 1)
        self._bi_carrying = None

        # SciWorld state
        self._sw_location = "chemistry_lab"
        self._sw_inventory: list[str] = []

        # WebArena state
        self._wa_url = "https://shop.example.com"

    def get_tools_schema(self) -> list[dict]:
        dispatch = {
            "webarena": self._webarena_tools,
            "searchqa": self._searchqa_tools,
            "textcraft": self._textcraft_tools,
            "babyai": self._babyai_tools,
            "sciworld": self._sciworld_tools,
        }
        return dispatch[self.environment]()

    # ---- WebArena tools ----

    def _webarena_tools(self) -> list[dict]:
        return [
            _tool("click", "Click an element by its numeric id from the accessibility tree.", {"element_id": "integer"}),
            _tool("type", "Type text into an input element.", {"element_id": "integer", "text": "string"}),
            _tool("goto", "Navigate to a URL.", {"url": "string"}),
            _tool("scroll", "Scroll the page.", {"direction": "string"}),
            _tool("go_back", "Go back to the previous page.", {}),
            _tool("search", "Use the site search.", {"query": "string"}),
        ]

    # ---- SearchQA tools ----

    def _searchqa_tools(self) -> list[dict]:
        return [
            _tool("search", "Search for information.", {"query": "string"}),
            _tool("read_document", "Read a document by its URL or ID.", {"doc_id": "string"}),
            _tool("submit_answer", "Submit your final answer.", {"answer": "string"}),
        ]

    # ---- TextCraft tools ----

    def _textcraft_tools(self) -> list[dict]:
        return [
            _tool("get", "Pick up an item from the environment.", {"item": "string"}),
            _tool("craft", "Craft an item using materials in your inventory.", {"item": "string"}),
            _tool("inventory", "Check your current inventory.", {}),
            _tool("look", "Look around the current location.", {}),
            _tool("move", "Move to an adjacent area.", {"direction": "string"}),
            _tool("recipes", "List available crafting recipes.", {}),
        ]

    # ---- BabyAI tools ----

    def _babyai_tools(self) -> list[dict]:
        return [
            _tool("move_forward", "Move one step forward.", {}),
            _tool("turn_left", "Turn 90 degrees left.", {}),
            _tool("turn_right", "Turn 90 degrees right.", {}),
            _tool("pick_up", "Pick up the object in front of you.", {}),
            _tool("drop", "Drop the object you are carrying.", {}),
            _tool("toggle", "Toggle/open the object in front of you.", {}),
            _tool("done", "Declare the task complete.", {}),
        ]

    # ---- SciWorld tools ----

    def _sciworld_tools(self) -> list[dict]:
        return [
            _tool("look", "Observe the current location.", {}),
            _tool("move_to", "Move to another location.", {"location": "string"}),
            _tool("pick_up", "Pick up an item.", {"item": "string"}),
            _tool("put_down", "Put down an item from inventory.", {"item": "string"}),
            _tool("pour", "Pour one item into another.", {"source": "string", "target": "string"}),
            _tool("heat", "Heat an item using a heat source.", {"item": "string"}),
            _tool("cool", "Cool an item.", {"item": "string"}),
            _tool("mix", "Mix two items together.", {"item1": "string", "item2": "string"}),
            _tool("measure", "Measure a property of an item.", {"item": "string"}),
            _tool("use", "Use one item on another.", {"tool_item": "string", "target": "string"}),
            _tool("done", "Declare the experiment complete and submit findings.", {"findings": "string"}),
        ]

    # ---- Execution dispatch ----

    async def execute(self, tool_name: str, args: dict) -> str:
        self._step += 1
        dispatch = {
            "webarena": self._exec_webarena,
            "searchqa": self._exec_searchqa,
            "textcraft": self._exec_textcraft,
            "babyai": self._exec_babyai,
            "sciworld": self._exec_sciworld,
        }
        handler = dispatch.get(self.environment)
        if handler is None:
            return f"Unknown environment: {self.environment}"
        return await handler(tool_name, args)

    async def _exec_webarena(self, tool_name: str, args: dict) -> str:
        if tool_name == "goto":
            self._wa_url = args.get("url", self._wa_url)
            return _gen_webarena_obs(self._wa_url, "goto")
        elif tool_name == "click":
            return _gen_webarena_obs(self._wa_url, f"click_{args.get('element_id', 0)}")
        elif tool_name == "type":
            text = args.get("text", "")
            return f"Typed '{text[:80]}' into element {args.get('element_id', 0)}.\n" + _gen_webarena_obs(
                self._wa_url, f"type_{text[:20]}"
            )
        elif tool_name == "scroll":
            return _gen_webarena_obs(self._wa_url, f"scroll_{args.get('direction', 'down')}")
        elif tool_name == "go_back":
            return _gen_webarena_obs(self._wa_url, "back")
        elif tool_name == "search":
            self._wa_url = f"https://shop.example.com/search?q={args.get('query', '')[:30]}"
            return _gen_webarena_obs(self._wa_url, "search")
        return f"Unknown WebArena action: {tool_name}"

    async def _exec_searchqa(self, tool_name: str, args: dict) -> str:
        if tool_name == "search":
            return _gen_search_results(args.get("query", ""), n=5)
        elif tool_name == "read_document":
            return _gen_document_content(args.get("doc_id", "default"))
        elif tool_name == "submit_answer":
            return f"Answer submitted: '{args.get('answer', '')[:200]}'. Task complete."
        return f"Unknown SearchQA action: {tool_name}"

    async def _exec_textcraft(self, tool_name: str, args: dict) -> str:
        if tool_name == "get":
            item = args.get("item", "wood_plank")
            rng = _deterministic_rng(item + str(self._step))
            if rng.random() < 0.8:
                self._tc_inventory[item] = self._tc_inventory.get(item, 0) + 1
                return f"Picked up {item}.\n" + _gen_textcraft_obs(self._tc_inventory, self._tc_location)
            return f"Could not find {item} nearby.\n" + _gen_textcraft_obs(self._tc_inventory, self._tc_location)

        elif tool_name == "craft":
            item = args.get("item", "")
            recipe = _TEXTCRAFT_RECIPES.get(item)
            if recipe is None:
                return f"No recipe for '{item}'. Use 'recipes' to see available recipes."
            missing = {}
            for mat in recipe:
                needed = missing.get(mat, 0) + 1
                missing[mat] = needed
            can_craft = all(self._tc_inventory.get(m, 0) >= c for m, c in missing.items())
            if can_craft:
                for m, c in missing.items():
                    self._tc_inventory[m] -= c
                self._tc_inventory[item] = self._tc_inventory.get(item, 0) + 1
                return f"Crafted {item}!\n" + _gen_textcraft_obs(self._tc_inventory, self._tc_location)
            return f"Missing materials for {item}. Need: {missing}\n" + _gen_textcraft_obs(
                self._tc_inventory, self._tc_location
            )

        elif tool_name == "inventory":
            return _gen_textcraft_obs(self._tc_inventory, self._tc_location)

        elif tool_name == "look":
            return _gen_textcraft_obs(self._tc_inventory, self._tc_location)

        elif tool_name == "move":
            direction = args.get("direction", "north")
            self._tc_location = f"{direction}_area"
            return f"Moved {direction}.\n" + _gen_textcraft_obs(self._tc_inventory, self._tc_location)

        elif tool_name == "recipes":
            lines = [f"  {name}: {', '.join(mats)}" for name, mats in _TEXTCRAFT_RECIPES.items()]
            return "Available recipes:\n" + "\n".join(lines)

        return f"Unknown TextCraft action: {tool_name}"

    async def _exec_babyai(self, tool_name: str, args: dict) -> str:
        x, y = self._bi_pos
        if tool_name == "move_forward":
            self._bi_pos = (x, min(y + 1, 7))
            return f"Moved forward.\n" + _gen_babyai_obs(self._bi_pos)
        elif tool_name == "turn_left":
            return f"Turned left.\n" + _gen_babyai_obs(self._bi_pos)
        elif tool_name == "turn_right":
            return f"Turned right.\n" + _gen_babyai_obs(self._bi_pos)
        elif tool_name == "pick_up":
            rng = _deterministic_rng(str(self._bi_pos) + str(self._step))
            obj = rng.choice(_BABYAI_COLORS) + " " + rng.choice(_BABYAI_OBJECTS)
            self._bi_carrying = obj
            return f"Picked up {obj}.\n" + _gen_babyai_obs(self._bi_pos)
        elif tool_name == "drop":
            dropped = self._bi_carrying or "nothing"
            self._bi_carrying = None
            return f"Dropped {dropped}.\n" + _gen_babyai_obs(self._bi_pos)
        elif tool_name == "toggle":
            return "Toggled the object in front.\n" + _gen_babyai_obs(self._bi_pos)
        elif tool_name == "done":
            return "Task declared complete."
        return f"Unknown BabyAI action: {tool_name}"

    async def _exec_sciworld(self, tool_name: str, args: dict) -> str:
        if tool_name == "look":
            return _gen_sciworld_obs(self._sw_location, self._sw_inventory, self._step)

        elif tool_name == "move_to":
            loc = args.get("location", "chemistry_lab")
            if loc in _SCIWORLD_LOCATIONS:
                self._sw_location = loc
                return f"Moved to {loc}.\n" + _gen_sciworld_obs(self._sw_location, self._sw_inventory, self._step)
            return f"Cannot move to '{loc}'. Available: {', '.join(_SCIWORLD_LOCATIONS)}"

        elif tool_name == "pick_up":
            item = args.get("item", "beaker")
            self._sw_inventory.append(item)
            return f"Picked up {item}.\n" + _gen_sciworld_obs(self._sw_location, self._sw_inventory, self._step)

        elif tool_name == "put_down":
            item = args.get("item", "")
            if item in self._sw_inventory:
                self._sw_inventory.remove(item)
                return f"Put down {item}.\n" + _gen_sciworld_obs(self._sw_location, self._sw_inventory, self._step)
            return f"You don't have {item}."

        elif tool_name in ("pour", "heat", "cool", "mix", "measure", "use"):
            rng = _deterministic_rng(tool_name + str(args) + str(self._step))
            observations = [
                "The solution changes color to a pale yellow.",
                "Temperature reading: 78.3°C. The substance begins to bubble.",
                "pH measurement: 6.2 (slightly acidic).",
                "The mixture produces a mild exothermic reaction.",
                "Mass measurement: 45.7g. No visible change observed.",
                "The sample dissolves completely after 30 seconds of stirring.",
                "A precipitate forms at the bottom of the beaker.",
                "The indicator turns blue, suggesting a basic solution.",
            ]
            result = rng.choice(observations)
            return f"[{tool_name}] {result}\n" + _gen_sciworld_obs(self._sw_location, self._sw_inventory, self._step)

        elif tool_name == "done":
            findings = args.get("findings", "No findings submitted.")
            return f"Experiment complete. Findings recorded:\n{findings}"

        return f"Unknown SciWorld action: {tool_name}"


# ---------------------------------------------------------------------------
# Helper to build OpenAI-style tool schema entries
# ---------------------------------------------------------------------------

def _tool(name: str, description: str, params: dict) -> dict:
    properties = {}
    required = []
    for pname, ptype in params.items():
        if ptype == "integer":
            properties[pname] = {"type": "integer", "description": pname.replace("_", " ").title()}
        else:
            properties[pname] = {"type": "string", "description": pname.replace("_", " ").title()}
        required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
