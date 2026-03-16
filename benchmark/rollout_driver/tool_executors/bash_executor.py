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
Bash executor for SWE-Agent style tool calling.
Executes bash commands in a subprocess with timeout.
"""

import asyncio
import os
import tempfile


class BashExecutor:
    def __init__(self, workdir: str | None = None, timeout: int = 30, **kwargs):
        self.workdir = workdir or tempfile.mkdtemp(prefix="swe_agent_")
        self.timeout = timeout
        os.makedirs(self.workdir, exist_ok=True)

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command in the working directory. Use for file exploration, editing, running tests, git operations, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to working directory."},
                            "content": {"type": "string", "description": "Content to write."},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content of a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to working directory."},
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return await self._run_bash(args.get("command", "echo 'no command'"))
        elif tool_name == "write_file":
            return self._write_file(args.get("path", ""), args.get("content", ""))
        elif tool_name == "read_file":
            return self._read_file(args.get("path", ""))
        else:
            return f"Unknown tool: {tool_name}"

    async def _run_bash(self, command: str) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workdir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            output = ""
            if stdout:
                output += stdout.decode(errors="replace")
            if stderr:
                output += "\nSTDERR:\n" + stderr.decode(errors="replace")
            if proc.returncode != 0:
                output += f"\n[exit code: {proc.returncode}]"
            # Truncate very long output
            if len(output) > 8000:
                output = output[:4000] + "\n...[truncated]...\n" + output[-4000:]
            return output or "(no output)"
        except asyncio.TimeoutError:
            return f"[command timed out after {self.timeout}s]"
        except Exception as e:
            return f"[error: {str(e)[:500]}]"

    def _write_file(self, path: str, content: str) -> str:
        full_path = os.path.join(self.workdir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"

    def _read_file(self, path: str) -> str:
        full_path = os.path.join(self.workdir, path)
        try:
            with open(full_path) as f:
                content = f.read()
            if len(content) > 8000:
                content = content[:4000] + "\n...[truncated]...\n" + content[-4000:]
            return content
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"
