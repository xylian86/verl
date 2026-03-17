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

Output size is controlled by max_output_chars (default 64000) to allow
large schema metadata responses that drive realistic token accumulation
in multi-turn rollouts.
"""

import json
import os
import sqlite3
import tempfile

DEFAULT_MAX_OUTPUT = 64000


class SQLiteExecutor:
    def __init__(self, db_path: str | None = None, max_output_chars: int = DEFAULT_MAX_OUTPUT, **kwargs):
        self.max_output = max_output_chars
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
                    "description": "List all tables and views in the database. Optionally filter by a prefix or pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prefix": {"type": "string", "description": "Optional prefix to filter tables (e.g. 'dim_', 'fact_', 'hr_')."},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_table",
                    "description": "Get the schema of a specific table including column names, types, and row count.",
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
                    "name": "search_tables",
                    "description": "Search for tables whose names or column names match a keyword. Useful for schema discovery in large databases.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "Keyword to search for in table and column names (case-insensitive)."},
                        },
                        "required": ["keyword"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_columns",
                    "description": "Search for columns across all tables matching a keyword. Returns table name, column name, and type.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "Keyword to search for in column names (case-insensitive)."},
                        },
                        "required": ["keyword"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_sample_data",
                    "description": "Get a sample of rows from a table. Useful for understanding the data before writing queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Name of the table to sample."},
                            "limit": {"type": "integer", "description": "Number of rows to return (default 5, max 20)."},
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
            return self._list_tables(args.get("prefix"))
        elif tool_name == "describe_table":
            return self._describe_table(args.get("table_name", ""))
        elif tool_name == "search_tables":
            return self._search_tables(args.get("keyword", ""))
        elif tool_name == "search_columns":
            return self._search_columns(args.get("keyword", ""))
        elif tool_name == "get_sample_data":
            return self._get_sample_data(args.get("table_name", ""), args.get("limit", 5))
        elif tool_name == "execute_python":
            return await self._run_python(args.get("code", ""))
        return f"Unknown tool: {tool_name}"

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_output:
            half = self.max_output // 2
            return text[:half] + f"\n...[truncated {len(text) - self.max_output} chars]...\n" + text[-half:]
        return text

    def _execute_sql(self, query: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                max_rows = 200
                if len(rows) > max_rows:
                    result_rows = [dict(r) for r in rows[:max_rows]]
                    result = json.dumps({"columns": columns, "rows": result_rows,
                                         "total_rows": len(rows), "showing": max_rows, "truncated": True}, indent=2)
                else:
                    result_rows = [dict(r) for r in rows]
                    result = json.dumps({"columns": columns, "rows": result_rows, "total_rows": len(rows)}, indent=2)
            else:
                conn.commit()
                result = f"Query executed successfully. Rows affected: {cursor.rowcount}"
            conn.close()
            return self._truncate(result)
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def _list_tables(self, prefix: str | None = None) -> str:
        """List tables with column count and row count to give the agent
        enough context to decide which tables to investigate further."""
        try:
            conn = sqlite3.connect(self.db_path)
            if prefix:
                cursor = conn.execute(
                    "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name LIKE ? ORDER BY name",
                    (prefix + "%",))
            else:
                cursor = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")
            raw_tables = cursor.fetchall()

            objects = []
            for tname, ttype in raw_tables:
                entry = {"name": tname, "type": ttype}
                try:
                    cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
                    entry["columns"] = len(cols)
                    entry["column_names"] = [c[1] for c in cols]
                    if ttype == "table":
                        entry["row_count"] = conn.execute(f"SELECT COUNT(*) FROM \"{tname}\"").fetchone()[0]
                except Exception:
                    pass
                objects.append(entry)

            conn.close()
            summary = f"Found {len(objects)} objects in database"
            if prefix:
                summary += f" matching prefix '{prefix}'"
            summary += ". Use describe_table for full schema details, get_sample_data to preview rows."
            return self._truncate(json.dumps({"summary": summary, "objects": objects}, indent=2))
        except Exception as e:
            return f"Error: {str(e)}"

    def _describe_table(self, table_name: str) -> str:
        """Return full schema with column statistics (distinct count, nulls,
        min/max, sample values) to make each describe_table call information-rich."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            if not columns:
                conn.close()
                return f"Error: Table '{table_name}' not found. Use list_tables or search_tables to find valid table names."
            row_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]

            col_details = []
            for c in columns:
                col_name = c[1]
                detail = {
                    "cid": c[0], "name": col_name, "type": c[2],
                    "notnull": c[3], "default": c[4], "pk": c[5],
                }
                if row_count > 0:
                    try:
                        stats = conn.execute(
                            f"SELECT COUNT(DISTINCT \"{col_name}\") AS distinct_count, "
                            f"SUM(CASE WHEN \"{col_name}\" IS NULL THEN 1 ELSE 0 END) AS null_count, "
                            f"MIN(\"{col_name}\") AS min_val, MAX(\"{col_name}\") AS max_val "
                            f"FROM \"{table_name}\""
                        ).fetchone()
                        detail["distinct_count"] = stats[0]
                        detail["null_count"] = stats[1]
                        detail["min"] = stats[2]
                        detail["max"] = stats[3]
                        # Sample values (up to 5 distinct)
                        samples = conn.execute(
                            f"SELECT DISTINCT \"{col_name}\" FROM \"{table_name}\" "
                            f"WHERE \"{col_name}\" IS NOT NULL LIMIT 5"
                        ).fetchall()
                        detail["sample_values"] = [s[0] for s in samples]
                    except Exception:
                        pass
                col_details.append(detail)

            # Foreign keys
            fk_cursor = conn.execute(f"PRAGMA foreign_key_list('{table_name}')")
            fks = [{"from": fk[3], "to_table": fk[2], "to_column": fk[4]} for fk in fk_cursor.fetchall()]
            # Indexes
            idx_cursor = conn.execute(f"PRAGMA index_list('{table_name}')")
            indexes = [{"name": idx[1], "unique": bool(idx[2])} for idx in idx_cursor.fetchall()]
            # CREATE TABLE statement
            create_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table_name,)
            ).fetchone()
            conn.close()

            result = {"table": table_name, "row_count": row_count, "columns": col_details}
            if fks:
                result["foreign_keys"] = fks
            if indexes:
                result["indexes"] = indexes
            if create_sql and create_sql[0]:
                result["create_statement"] = create_sql[0]
            return self._truncate(json.dumps(result, indent=2))
        except Exception as e:
            return f"Error: {str(e)}"

    def _search_tables(self, keyword: str) -> str:
        """Search tables/views by name or column name. Returns full column
        lists for matching tables so the agent can assess relevance."""
        if not keyword:
            return "Error: keyword is required"
        try:
            conn = sqlite3.connect(self.db_path)
            kw = keyword.lower()
            name_matches = conn.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND LOWER(name) LIKE ? ORDER BY name",
                (f"%{kw}%",)).fetchall()
            name_results = []
            for tname, ttype in name_matches:
                entry = {"name": tname, "type": ttype}
                try:
                    cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
                    entry["columns"] = [{"name": c[1], "type": c[2]} for c in cols]
                    if ttype == "table":
                        entry["row_count"] = conn.execute(f"SELECT COUNT(*) FROM \"{tname}\"").fetchone()[0]
                except Exception:
                    pass
                name_results.append(entry)

            all_tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            col_matches = []
            for (tname,) in all_tables:
                if any(tname == nm[0] for nm in name_matches):
                    continue
                cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
                matching_cols = [{"name": c[1], "type": c[2]} for c in cols if kw in c[1].lower()]
                if matching_cols:
                    try:
                        rc = conn.execute(f"SELECT COUNT(*) FROM \"{tname}\"").fetchone()[0]
                    except Exception:
                        rc = None
                    col_matches.append({"table": tname, "matching_columns": matching_cols,
                                        "all_columns": [c[1] for c in cols], "row_count": rc})
            conn.close()
            result = {
                "keyword": keyword,
                "tables_matching_by_name": name_results,
                "tables_matching_by_column": col_matches[:80],
            }
            return self._truncate(json.dumps(result, indent=2))
        except Exception as e:
            return f"Error: {str(e)}"

    def _search_columns(self, keyword: str) -> str:
        """Search for columns across all tables matching a keyword."""
        if not keyword:
            return "Error: keyword is required"
        try:
            conn = sqlite3.connect(self.db_path)
            kw = keyword.lower()
            all_tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            matches = []
            for (tname,) in all_tables:
                cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
                for c in cols:
                    if kw in c[1].lower():
                        entry = {"table": tname, "column": c[1], "type": c[2], "pk": bool(c[5])}
                        try:
                            distinct = conn.execute(
                                f"SELECT COUNT(DISTINCT \"{c[1]}\") FROM \"{tname}\""
                            ).fetchone()[0]
                            entry["distinct_values"] = distinct
                        except Exception:
                            pass
                        matches.append(entry)
            conn.close()
            return self._truncate(json.dumps(
                {"keyword": keyword, "total_matches": len(matches), "matches": matches}, indent=2))
        except Exception as e:
            return f"Error: {str(e)}"

    def _get_sample_data(self, table_name: str, limit: int = 10) -> str:
        """Get sample rows from a table."""
        limit = min(max(1, limit), 20)
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"SELECT * FROM \"{table_name}\" LIMIT ?", (limit,))
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                rows = [dict(r) for r in cursor.fetchall()]
                total = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
                conn.close()
                result = json.dumps({"table": table_name, "columns": columns,
                                     "sample_rows": rows, "total_rows": total}, indent=2)
                return self._truncate(result)
            conn.close()
            return f"No data in table '{table_name}'."
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
            return self._truncate(output) if output else "(no output)"
        except asyncio.TimeoutError:
            return "[execution timed out after 30s]"
        except Exception as e:
            return f"[error: {str(e)[:500]}]"
