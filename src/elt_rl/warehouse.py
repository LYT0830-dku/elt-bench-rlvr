"""One warehouse contract for tool execution and host-side grading."""

from __future__ import annotations

import math
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import duckdb

from elt_rl.errors import InfrastructureError
from elt_rl.sql_policy import validate
from elt_rl.task import identifier


@dataclass(frozen=True)
class TableData:
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    truncated: bool = False

    def public(self):
        return {
            "columns": self.columns,
            "rows": tuple(
                tuple(str(v) if isinstance(v, float) and not math.isfinite(v) else v for v in row)
                for row in self.rows
            ),
            "truncated": self.truncated,
        }


def policy_error(exc):
    if isinstance(exc, (ValueError, duckdb.ProgrammingError, duckdb.ConversionException)):
        return True
    if type(exc).__module__.startswith("snowflake.connector"):
        return str(getattr(exc, "sqlstate", ""))[:2] not in {"08", "28"} and type(exc).__name__ in {
            "ProgrammingError",
            "DataError",
            "IntegrityError",
        }
    return False


class Warehouse(ABC):
    dialect: str
    schema: str
    database: str | None = None

    def execute(self, sql, *, limit=100):
        tree = validate(sql, self.dialect, schemas={"airbyte_schema"})
        from sqlglot import exp

        for table in tree.find_all(exp.Table):
            if table.db:
                table.set(
                    "db",
                    exp.to_identifier(
                        self.schema.upper() if self.dialect == "snowflake" else self.schema,
                        quoted=True,
                    ),
                )
        if self.dialect == "snowflake":
            for node in tree.find_all(exp.Identifier):
                node.set("this", node.this.upper())
                node.set("quoted", True)
        return self._query(tree.sql(dialect=self.dialect), limit)

    def quote(self, name):
        return (
            '"'
            + (identifier(name).upper() if self.dialect == "snowflake" else identifier(name))
            + '"'
        )

    def output(self, name, limit=2_000_000):
        return self._query(f"SELECT * FROM {self.quote(self.schema)}.{self.quote(name)}", limit)

    def prune_relations(self, keep):
        """Remove obsolete objects only inside this episode's assigned schema."""
        schema = self.schema.upper() if self.dialect == "snowflake" else self.schema
        rows = self._query(
            "SELECT table_name, table_type FROM information_schema.tables "
            + "WHERE table_schema = '"
            + identifier(schema)
            + "'",
            10000,
        )
        if rows.truncated:
            raise InfrastructureError("Cannot reconcile a truncated warehouse inventory")
        removed = []
        for name, kind in sorted(rows.rows, key=lambda row: row[1] != "VIEW"):
            if name.lower() in keep:
                continue
            object_type = "VIEW" if kind == "VIEW" else "TABLE"
            self._query(
                f"DROP {object_type} IF EXISTS {self.quote(self.schema)}.{self.quote(name.lower())}",
                0,
            )
            removed.append(name.lower())
        return removed

    @abstractmethod
    def _query(self, sql: str, limit: int) -> TableData: ...
    @abstractmethod
    def close(self): ...


class DuckDBWarehouse(Warehouse):
    dialect = "duckdb"

    def __init__(self, path: Path, schema="airbyte_schema", timeout=None):
        self.path = Path(path)
        self.schema = identifier(schema)
        self.timeout = timeout
        self._query(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"', 0)

    def _query(self, sql, limit):
        # dbt is a separate process: never retain a competing file lock across tools.
        with duckdb.connect(
            str(self.path),
            config={"enable_external_access": "false", "threads": 1, "memory_limit": "512MB"},
        ) as conn:
            timer = (
                threading.Timer(self.timeout, conn.interrupt) if self.timeout is not None else None
            )
            if timer is not None:
                timer.daemon = True
                timer.start()
            try:
                cursor = conn.execute(sql)
                columns = tuple(d[0].lower() for d in cursor.description or [])
                rows = cursor.fetchmany(limit + 1) if limit else []
                return TableData(columns, tuple(rows[:limit]), len(rows) > limit)
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join()

    def close(self):
        pass


class SnowflakeWarehouse(Warehouse):
    dialect = "snowflake"

    def __init__(self, config: dict, admin_config: dict, on_resources=None):
        import snowflake.connector

        if not admin_config:
            raise InfrastructureError(
                "Native isolation requires host-only grant-manager credentials"
            )
        self.config = dict(config)
        self.database = identifier(config["database"]).upper()
        self.schema = "RL_" + uuid.uuid4().hex.upper()
        self.internal_schema = self.schema + "_INTERNAL"
        self.role = self.schema + "_ROLE"
        self.config["role"] = self.role
        self.created = []
        self.role_created = False
        self.schema_creation_pending = []
        self.role_creation_pending = False
        self.conn = None
        self.on_resources = on_resources
        self.admin = snowflake.connector.connect(
            **admin_config,
            session_parameters={
                "QUERY_TAG": "elt-rl/resource-manager",
            },
        )

        def execute(query):
            with self.admin.cursor() as cursor:
                cursor.execute(query)

        self._admin_execute = execute
        try:
            self._record("allocating")
            self.role_creation_pending = True
            self._record("role_creation_pending")
            execute(f'CREATE ROLE "{self.role}"')
            self.role_created = True
            self.role_creation_pending = False
            self._record("role_created")
            execute(f'GRANT USAGE ON DATABASE "{self.database.upper()}" TO ROLE "{self.role}"')
            execute(
                f'GRANT USAGE ON WAREHOUSE "{identifier(config["warehouse"]).upper()}" TO ROLE "{self.role}"'
            )
            for schema in [self.schema, self.internal_schema]:
                self.schema_creation_pending.append(schema)
                self._record("schema_creation_pending")
                execute(f'CREATE SCHEMA "{self.database.upper()}"."{schema}"')
                self.created.append(schema)
                self.schema_creation_pending.remove(schema)
                self._record("schema_created")
                execute(
                    f'GRANT ALL PRIVILEGES ON SCHEMA "{self.database.upper()}"."{schema}" TO ROLE "{self.role}"'
                )
            execute(f'GRANT ROLE "{self.role}" TO USER "{identifier(config["user"]).upper()}"')
            self.conn = snowflake.connector.connect(
                **self.config,
                session_parameters={
                    "QUERY_TAG": "elt-bench-rl/native",
                },
            )
            with self.conn.cursor() as cursor:
                cursor.execute("USE SECONDARY ROLES NONE")
            self._record("allocated")
        except BaseException:
            self.close()
            raise

    def _record(self, status):
        if self.on_resources:
            self.on_resources(
                {
                    "database": self.database,
                    "schema": self.schema,
                    "internal_schema": self.internal_schema,
                    "role": self.role,
                    "remaining_schemas": list(
                        dict.fromkeys(self.created + self.schema_creation_pending)
                    ),
                    "schema_creation_pending": list(self.schema_creation_pending),
                    "role_exists": self.role_created,
                    "role_creation_pending": self.role_creation_pending,
                    "state": status,
                }
            )

    def _query(self, sql, limit):
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchmany(limit + 1) if limit else []
            return TableData(
                tuple(d[0].lower() for d in cursor.description or []),
                tuple(rows[:limit]),
                len(rows) > limit,
            )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
        failures = []
        for schema in list(dict.fromkeys(self.created + self.schema_creation_pending)):
            for attempt in range(3):
                try:
                    self._admin_execute(
                        f'DROP SCHEMA IF EXISTS "{self.database.upper()}"."{schema}" CASCADE'
                    )
                    if schema in self.created:
                        self.created.remove(schema)
                    break
                except Exception:
                    if attempt == 2:
                        failures.append(schema)
                    else:
                        time.sleep(0.1 * (attempt + 1))
        if (self.role_created or self.role_creation_pending) and not failures:
            for attempt in range(3):
                try:
                    self._admin_execute(f'DROP ROLE IF EXISTS "{self.role}"')
                    self.role_created = False
                    break
                except Exception:
                    if attempt == 2:
                        failures.append(self.role)
                    else:
                        time.sleep(0.1 * (attempt + 1))
        # Removal does not settle an earlier CREATE whose reply was lost.
        failures.extend(self.schema_creation_pending)
        if self.role_creation_pending:
            failures.append(self.role)
        self._record("cleanup_failed" if failures else "closed")
        if not failures:
            self.admin.close()
        if failures:
            raise InfrastructureError(
                "Warehouse resources require reconciliation: " + str(failures)
            )
