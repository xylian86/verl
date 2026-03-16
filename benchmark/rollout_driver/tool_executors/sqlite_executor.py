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
SQLite executor for NL2SQL and data pipeline debugging tasks.
Operates on a pre-created SQLite database.
"""

import json
import os
import sqlite3
import tempfile


class SQLiteExecutor:
    def __init__(self, db_path: str | None = None, **kwargs):
        if db_path and os.path.exists(db_path):
            self.db_path = db_path
        else:
            self.db_path = os.path.join(tempfile.mkdtemp(prefix="sqlite_"), "benchmark.db")
            self._create_default_db()

    def _create_default_db(self):
        """Create a minimal default database if none provided."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT OR IGNORE INTO test VALUES (1, 'default')")
        conn.commit()
        conn.close()

    def get_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SQL query against the database and return results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query to execute."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_tables",
                    "description": "List all tables in the database.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_table",
                    "description": "Get the schema of a specific table.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Name of the table."},
                        },
                        "required": ["table_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code for data analysis. The database is available at the path stored in variable DB_PATH.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute."},
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

    async def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "execute_sql":
            return self._execute_sql(args.get("query", ""))
        elif tool_name == "list_tables":
            return self._list_tables()
        elif tool_name == "describe_table":
            return self._describe_table(args.get("table_name", ""))
        elif tool_name == "execute_python":
            return await self._run_python(args.get("code", ""))
        return f"Unknown tool: {tool_name}"

    def _execute_sql(self, query: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                if len(rows) > 100:
                    result_rows = [dict(r) for r in rows[:100]]
                    result = json.dumps({"columns": columns, "rows": result_rows, "total_rows": len(rows), "truncated": True}, indent=2)
                else:
                    result_rows = [dict(r) for r in rows]
                    result = json.dumps({"columns": columns, "rows": result_rows, "total_rows": len(rows)}, indent=2)
            else:
                conn.commit()
                result = f"Query executed successfully. Rows affected: {cursor.rowcount}"
            conn.close()
            if len(result) > 8000:
                result = result[:4000] + "\n...[truncated]...\n" + result[-4000:]
            return result
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def _list_tables(self) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")
            tables = cursor.fetchall()
            conn.close()
            return json.dumps([{"name": t[0], "type": t[1]} for t in tables], indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _describe_table(self, table_name: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            # Also get row count
            count = conn.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]
            conn.close()
            schema = [
                {"cid": c[0], "name": c[1], "type": c[2], "notnull": c[3], "default": c[4], "pk": c[5]}
                for c in columns
            ]
            return json.dumps({"table": table_name, "columns": schema, "row_count": count}, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    async def _run_python(self, code: str) -> str:
        import asyncio

        # Inject DB_PATH into the code
        wrapped = f'DB_PATH = "{self.db_path}"\n{code}'
        code_file = os.path.join(os.path.dirname(self.db_path), "_exec.py")
        with open(code_file, "w") as f:
            f.write(wrapped)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = ""
            if stdout:
                output += stdout.decode(errors="replace")
            if stderr:
                output += "\nSTDERR:\n" + stderr.decode(errors="replace")
            if len(output) > 8000:
                output = output[:4000] + "\n...[truncated]...\n" + output[-4000:]
            return output or "(no output)"
        except asyncio.TimeoutError:
            return "[execution timed out after 30s]"
        except Exception as e:
            return f"[error: {str(e)[:500]}]"
