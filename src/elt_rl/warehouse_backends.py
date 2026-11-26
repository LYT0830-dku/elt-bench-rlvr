"""Warehouse-specific connection, loading, transformation and cleanup adapters."""

from __future__ import annotations

from pathlib import Path

from elt_rl.registry import Registry
from elt_rl.warehouse import DuckDBWarehouse, SnowflakeWarehouse


class WarehouseBackend:
    remote = False

    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.on_resources = None

    @property
    def dialect(self):
        return self.warehouse.dialect

    def query(self, sql, *, limit=100):
        return self.warehouse.execute(sql, limit=limit)

    def output(self, table, limit=2_000_000):
        return self.warehouse.output(table, limit)

    def table_exists(self, table):
        warehouse = self.warehouse
        schema = warehouse.schema.upper() if self.dialect == "snowflake" else warehouse.schema
        return bool(
            warehouse._query(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = '"
                + schema
                + "' AND LOWER(table_name) = '"
                + table
                + "'",
                1,
            ).rows
        )

    def physical_types(self, table):
        warehouse = self.warehouse
        qualified = warehouse.quote(warehouse.schema) + "." + warehouse.quote(table)
        statement, name_field, type_field = self.describe_protocol
        result = warehouse._query(statement + qualified, 100)
        return {
            str(row[result.columns.index(name_field)]).lower(): row[
                result.columns.index(type_field)
            ]
            for row in result.rows
        }

    def load(self, table, columns, types, values, mode):
        raise NotImplementedError

    def profile(self):
        raise NotImplementedError

    def transformation(self, runner, command, project, profiles):
        return runner.dbt(command, project, profiles, self.warehouse.schema)

    def cleanup(self):
        self.warehouse.close()

    def creation_sql(self, table, columns, types, mode):
        warehouse = self.warehouse
        qualified = warehouse.quote(warehouse.schema) + "." + warehouse.quote(table)
        definitions = ",".join(
            warehouse.quote(column) + " " + self.data_type(kind)
            for column, kind in zip(columns, types, strict=True)
        )
        create = (
            (
                "CREATE OR REPLACE TABLE "
                if mode == "full_refresh_overwrite"
                else "CREATE TABLE IF NOT EXISTS "
            )
            + qualified
            + " ("
            + definitions
            + ")"
        )
        return qualified, create

    def data_type(self, kind):
        return kind


class DuckDBBackend(WarehouseBackend):
    describe_protocol = ("DESCRIBE ", "column_name", "column_type")

    @classmethod
    def create(cls, root, configuration, on_resources=None):
        backend = cls(
            DuckDBWarehouse(
                Path(root) / "warehouse.duckdb",
                schema=configuration.get("logical_schema", "airbyte_schema"),
            )
        )
        backend.on_resources = on_resources
        return backend

    def cleanup(self):
        self.warehouse.close()
        if self.on_resources:
            self.on_resources(
                {
                    "state": "closed",
                    "backend": "duckdb",
                    "owned_path": str(self.warehouse.path),
                    "active_connections": 0,
                }
            )

    def load(self, table, columns, types, values, mode):
        import duckdb

        qualified, create = self.creation_sql(table, columns, types, mode)
        with duckdb.connect(
            str(self.warehouse.path),
            config={"enable_external_access": "false", "threads": 1, "memory_limit": "512MB"},
        ) as connection:
            connection.execute("BEGIN")
            try:
                connection.execute(create)
                if values:
                    connection.executemany(
                        "INSERT INTO "
                        + qualified
                        + " ("
                        + ",".join(self.warehouse.quote(c) for c in columns)
                        + ") VALUES ("
                        + ",".join("?" for _ in columns)
                        + ")",
                        values,
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def profile(self):
        return {
            "type": "duckdb",
            "path": str(self.warehouse.path),
            "schema": self.warehouse.schema,
            "threads": 1,
            "settings": {"enable_external_access": False, "memory_limit": "512MB"},
        }


class SnowflakeBackend(WarehouseBackend):
    remote = True
    describe_protocol = ("DESC TABLE ", "name", "type")

    @classmethod
    def create(cls, root, configuration, on_resources=None):
        return cls(
            SnowflakeWarehouse(
                configuration["snowflake"],
                configuration["snowflake_admin"],
                on_resources=on_resources,
            )
        )

    def data_type(self, kind):
        return "FLOAT" if kind in {"DOUBLE", "REAL"} else kind

    def load(self, table, columns, types, values, mode):
        qualified, create = self.creation_sql(table, columns, types, mode)
        with self.warehouse.conn.cursor() as cursor:
            cursor.execute(create)
            if values:
                cursor.executemany(
                    "INSERT INTO "
                    + qualified
                    + " ("
                    + ",".join(self.warehouse.quote(c) for c in columns)
                    + ") VALUES ("
                    + ",".join("%s" for _ in columns)
                    + ")",
                    values,
                )

    def profile(self):
        target = {
            key: value
            for key, value in self.warehouse.config.items()
            if key
            in {
                "account",
                "user",
                "password",
                "role",
                "database",
                "warehouse",
                "private_key_passphrase",
            }
        }
        target.update(type="snowflake", schema=self.warehouse.schema.upper(), threads=1)
        if self.warehouse.config.get("private_key_file"):
            target["private_key_path"] = self.warehouse.config["private_key_file"]
        return target

    def transformation(self, runner, command, project, profiles):
        return runner.dbt(
            command, project, profiles, self.warehouse.schema.upper(), role=self.warehouse.role
        )


WAREHOUSES = Registry("warehouse backend")
WAREHOUSES.register("duckdb", DuckDBBackend)
WAREHOUSES.register("snowflake", SnowflakeBackend)
