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
Code interpreter executor for math/code tasks.
Executes Python code in a subprocess.
"""

import asyncio
import os
import tempfile


class CodeInterpreterExecutor:
    def __init__(self, timeout: int = 30, **kwargs):
        self.timeout = timeout
        self.workdir = tempfile.mkdtemp(prefix="code_interp_")

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code and return the output. Use for calculations, data analysis, plotting, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "execute_python":
            return await self._run_python(args.get("code", "print('no code')"))
        return f"Unknown tool: {tool_name}"

    async def _run_python(self, code: str) -> str:
        # Write code to temp file and execute
        code_file = os.path.join(self.workdir, "_exec.py")
        with open(code_file, "w") as f:
            f.write(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                code_file,
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
            if len(output) > 8000:
                output = output[:4000] + "\n...[truncated]...\n" + output[-4000:]
            return output or "(no output)"
        except asyncio.TimeoutError:
            return f"[execution timed out after {self.timeout}s]"
        except Exception as e:
            return f"[error: {str(e)[:500]}]"
