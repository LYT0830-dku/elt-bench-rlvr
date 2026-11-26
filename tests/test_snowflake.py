"""Mocked Snowflake SQL/ownership contract; no cloud connection is made."""

import pytest

from elt_rl.warehouse import SnowflakeWarehouse


@pytest.mark.parametrize("transient_drop", [False, True])
def test_snowflake_namespace_role_and_cleanup_contract(monkeypatch, transient_drop):
    import snowflake.connector as connector

    queries, connections, failed_drops = [], [], []

    class Cursor:
        description = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            queries.append(sql)
            if transient_drop and sql.startswith("DROP SCHEMA") and not failed_drops:
                failed_drops.append(sql)
                raise RuntimeError("Transient cleanup failure")

        def fetchmany(self, limit):
            return []

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    def connect(**kwargs):
        connections.append(kwargs)
        return Connection()

    monkeypatch.setattr(connector, "connect", connect)
    config = {
        "account": "example",
        "user": "runner",
        "database": "dedicated",
        "role": "old_role",
        "warehouse": "wh",
    }
    manager = {**config, "user": "manager", "role": "resource_manager"}
    resources = []
    a = SnowflakeWarehouse(
        config, manager, on_resources=lambda record: resources.append((record, len(queries)))
    )
    b = SnowflakeWarehouse(config, manager)
    assert resources[0][0]["state"] == "allocating" and resources[0][1] == 0
    assert {a.schema, a.internal_schema}.isdisjoint({b.schema, b.internal_schema})
    assert (
        a.role != b.role and connections[1]["role"] == a.role and connections[3]["role"] == b.role
    )
    a.execute("SELECT * FROM airbyte_schema.price LIMIT 1")
    assert any('"' + a.schema + '"' in query and "price" in query.lower() for query in queries)
    expected_schemas = [a.schema, a.internal_schema, b.schema, b.internal_schema]
    a.close()
    b.close()
    assert resources[-1][0]["state"] == "closed"
    assert resources[-1][0]["remaining_schemas"] == [] and not resources[-1][0]["role_exists"]
    for schema in expected_schemas:
        assert f'DROP SCHEMA IF EXISTS "DEDICATED"."{schema}" CASCADE' in queries
    assert f'DROP ROLE IF EXISTS "{a.role}"' in queries
    assert "STATEMENT_TIMEOUT_IN_SECONDS" not in connections[1]["session_parameters"]
    assert "STATEMENT_QUEUED_TIMEOUT_IN_SECONDS" not in connections[1]["session_parameters"]
