"""Each dbt connection must lose access to other concurrently granted roles."""

import json

import pytest

from elt_rl.native_dbt_entrypoint import guard_connection


@pytest.mark.parametrize(
    "role,secondary,accepted",
    [
        ("EPISODE_ROLE", '{"roles":"","value":"NONE"}', True),
        ("EPISODE_ROLE", '{"roles":"","value":""}', True),
        ("OTHER_ROLE", '{"roles":"","value":"NONE"}', False),
        ("EPISODE_ROLE", '{"roles":"OTHER_ROLE","value":"ALL"}', False),
        ("EPISODE_ROLE", '{"roles":"OTHER_ROLE","value":"NONE"}', False),
        ("EPISODE_ROLE", "{}", False),
    ],
)
def test_dbt_role_guard_fails_closed(tmp_path, role, secondary, accepted):
    commands = []

    class Connection:
        closed = False

        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            commands.append(sql)
            return self

        def fetchone(self):
            return role, secondary

        def close(self):
            self.closed = True

    connection, path = Connection(), tmp_path / "guard.jsonl"
    if accepted:
        assert guard_connection(connection, "EPISODE_ROLE", path) is connection
        assert json.loads(path.read_text())["role"] == "EPISODE_ROLE"
        assert not connection.closed
    else:
        with pytest.raises(RuntimeError, match="exclusive episode role"):
            guard_connection(connection, "EPISODE_ROLE", path)
        assert connection.closed and not path.exists()
    assert commands[0] == "USE SECONDARY ROLES NONE"
