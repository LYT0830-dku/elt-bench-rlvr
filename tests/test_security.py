"""Execution capabilities and physical namespace mapping."""

import duckdb
import pytest
from sqlglot.errors import ParseError, UnsupportedError

from elt_rl.sql_policy import validate
from elt_rl.warehouse import DuckDBWarehouse, TableData, Warehouse


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH '/tmp/secret.duckdb' AS secret",
        "COPY airbyte_schema.orders TO '/tmp/leak.csv'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "SET enable_external_access=true",
        "SELECT * FROM read_csv('/tmp/expected.csv')",
        "SELECT * FROM read_parquet('https://example.com/reference.parquet')",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM other.airbyte_schema.orders",
        "SELECT * FROM airbyte_schema.orders; SELECT 1",
        "CREATE OR REPLACE TABLE airbyte_schema.orders AS SELECT 1 AS x",
        "DELETE FROM airbyte_schema.orders",
        "UPDATE airbyte_schema.orders SET amount=1",
        "CREATE VIEW airbyte_schema.x AS SELECT 1",
        "SELECT getenv('TINKER_API_KEY')",
        "SELECT current_setting('home_directory')",
        "SELECT * FROM query('SELECT 1')",
        "SELECT * INTO airbyte_schema.x FROM airbyte_schema.orders",
        "SELECT * FROM orders",
        "CREATE TABLE airbyte_schema.x AS SELECT * FROM read_json('/etc/passwd')",
        "CREATE TABLE airbyte_schema.x AS SELECT * FROM warehouse.secret.table",
        "CALL SYSTEM$SEND_EMAIL('a','b','c','d')",
        "SELECT system$reference('table','secret','session','select')",
    ],
)
def test_forbidden_sql(sql):
    with pytest.raises((ValueError, ParseError, UnsupportedError)):
        validate(sql, "duckdb", schemas={"airbyte_schema"})


def test_duckdb_defense_in_depth(tmp_path):
    db = DuckDBWarehouse(tmp_path / "data.duckdb")
    try:
        with pytest.raises(duckdb.PermissionException):
            db._query("SELECT * FROM read_csv('/etc/passwd')", 10)
    finally:
        db.close()


def test_destination_mapping_and_cte_preserves_snowflake_case():
    queries = []

    class Probe(Warehouse):
        dialect = "snowflake"
        schema = "RL_123"

        def _query(self, sql, limit):
            queries.append(sql)
            return TableData((), ())

        def close(self):
            pass

    Probe().execute('WITH q AS (SELECT "ID" FROM airbyte_schema."PRICE") SELECT * FROM q')
    assert '"RL_123"' in queries[0]
    assert '"PRICE"' in queries[0] and '"ID"' in queries[0]
    assert '"price"' not in queries[0] and '"id"' not in queries[0]
