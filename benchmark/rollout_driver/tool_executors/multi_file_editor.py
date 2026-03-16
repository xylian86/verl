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
Multi-file editor executor for code generation tasks.
Provides a virtual filesystem for reading/writing multiple files
and running tests.
"""

import asyncio
import os
import tempfile


class MultiFileEditorExecutor:
    def __init__(self, project_dir: str | None = None, timeout: int = 30, **kwargs):
        self.project_dir = project_dir or tempfile.mkdtemp(prefix="codegen_")
        self.timeout = timeout
        os.makedirs(self.project_dir, exist_ok=True)

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path relative to project root. Default: '.'",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to project root."},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file, creating directories as needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to project root."},
                            "content": {"type": "string", "description": "File content."},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command in the project directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run."},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for a pattern across all files in the project.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Text pattern to search for."},
                            "file_pattern": {"type": "string", "description": "Glob pattern for file names (e.g., '*.py')."},
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "list_files":
            return self._list_files(args.get("path", "."))
        elif tool_name == "read_file":
            return self._read_file(args.get("path", ""))
        elif tool_name == "write_file":
            return self._write_file(args.get("path", ""), args.get("content", ""))
        elif tool_name == "run_command":
            return await self._run_command(args.get("command", "echo 'no command'"))
        elif tool_name == "search_files":
            return await self._search_files(args.get("pattern", ""), args.get("file_pattern", "*"))
        return f"Unknown tool: {tool_name}"

    def _list_files(self, path: str) -> str:
        full_path = os.path.join(self.project_dir, path)
        try:
            entries = []
            for entry in sorted(os.listdir(full_path)):
                entry_path = os.path.join(full_path, entry)
                if os.path.isdir(entry_path):
                    entries.append(f"  {entry}/")
                else:
                    size = os.path.getsize(entry_path)
                    entries.append(f"  {entry} ({size} bytes)")
            return "\n".join(entries) if entries else "(empty directory)"
        except FileNotFoundError:
            return f"Directory not found: {path}"

    def _read_file(self, path: str) -> str:
        full_path = os.path.join(self.project_dir, path)
        try:
            with open(full_path) as f:
                content = f.read()
            if len(content) > 8000:
                content = content[:4000] + "\n...[truncated]...\n" + content[-4000:]
            return content
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error: {e}"

    def _write_file(self, path: str, content: str) -> str:
        full_path = os.path.join(self.project_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"

    async def _run_command(self, command: str) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            output = ""
            if stdout:
                output += stdout.decode(errors="replace")
            if stderr:
                output += "\nSTDERR:\n" + stderr.decode(errors="replace")
            if proc.returncode != 0:
                output += f"\n[exit code: {proc.returncode}]"
            if len(output) > 8000:
                output = output[:4000] + "\n...[truncated]...\n" + output[-4000:]
            return output or "(no output)"
        except asyncio.TimeoutError:
            return f"[command timed out after {self.timeout}s]"
        except Exception as e:
            return f"[error: {str(e)[:500]}]"

    async def _search_files(self, pattern: str, file_pattern: str) -> str:
        try:
            cmd = f"grep -rn '{pattern}' --include='{file_pattern}' . 2>/dev/null | head -50"
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode(errors="replace") if stdout else "(no matches)"
            return output
        except Exception as e:
            return f"Search error: {e}"
