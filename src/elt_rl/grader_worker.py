"""Private child process for original grader imports, argv and working directory."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.warehouse import policy_error


def main():
    root = Path(sys.argv[1]).resolve()
    config = json.loads(Path(sys.argv[2]).read_text())
    os.chdir(root)
    sys.argv = ["eva_stage2.py", "--folder", "run"]
    queries = []
    errors = []

    def checked(operation, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            errors.append(
                {
                    "operation": operation,
                    "exception": type(exc).__name__,
                    "sqlstate": getattr(exc, "sqlstate", None),
                    "infrastructure": operation == "connect" or not policy_error(exc),
                }
            )
            raise

    import snowflake.connector

    original_connect = snowflake.connector.connect

    def connect(**ignored):
        return Connection()

    class Connection:
        def __init__(self):
            self.real = checked("connect", lambda: original_connect(**config["connection"]))

        def cursor(self):
            return Cursor(self)

        def close(self):
            self.real.close()

        def commit(self):
            pass

        def rollback(self):
            pass

    class Cursor:
        def __init__(self, connection):
            self.connection = connection
            self.real = checked("cursor", connection.real.cursor)
            self.description = None

        def execute(self, query, *args):
            task = re.escape(config["task"])
            schema = config["schema"].upper()
            q = re.sub(rf"(?i)\b{task}\.airbyte_schema\.", f'"{schema}".', query)
            q = re.sub(
                rf"(?i)\b{task}\.information_schema\.",
                f'"{config["database"].upper()}".information_schema.',
                q,
            )
            q = q.replace("table_schema = 'AIRBYTE_SCHEMA'", f"table_schema = '{schema}'")
            # The Stage 1 driver expects the logical schema label in its result.
            if re.search(r"(?i)SELECT\s+schema_name", q):
                q = q.replace(
                    "schema_name",
                    f"CASE WHEN schema_name='{schema}' THEN 'AIRBYTE_SCHEMA' ELSE schema_name END AS schema_name",
                    1,
                )
            queries.append({"original": query, "executed": q})
            checked("execute", self.real.execute, q, *args)
            self.description = self.real.description
            return self

        def fetchall(self):
            return checked("fetchall", self.real.fetchall)

        def fetchmany(self, *args):
            return checked("fetchmany", self.real.fetchmany, *args)

        def close(self):
            self.real.close()

    snowflake.connector.connect = connect

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        load("eva_stage1").evaluate_stage1("run", {})
        load("eva_stage2").evaluate_stage2("run", {})
    finally:
        write_json(root / "driver_queries.json", queries)
        write_json(root / "driver_errors.json", errors)


if __name__ == "__main__":
    main()
