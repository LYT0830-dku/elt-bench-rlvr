"""Trusted fixed dbt launcher: disable secondary roles on every Snowflake connection.

Executed by the pinned dbt venv interpreter. This file is host code, never a
model-authored script, and does not import the control environment's dependencies.
"""

import json
import os
import sys
import time


def guard_connection(connection, expected_role, receipt):
    try:
        with connection.cursor() as cursor:
            cursor.execute("USE SECONDARY ROLES NONE")
            role, secondary = cursor.execute(
                "SELECT CURRENT_ROLE(), CURRENT_SECONDARY_ROLES()"
            ).fetchone()
        parsed = json.loads(secondary) if isinstance(secondary, str) else secondary
        if (
            role != expected_role
            or not isinstance(parsed, dict)
            or parsed.get("roles") != ""
            or parsed.get("value") not in {"", "NONE"}
        ):
            raise RuntimeError("dbt connection did not retain its exclusive episode role")
        event = {
            "time": time.time(),
            "role": role,
            "secondary_roles": parsed,
            "secondary_roles_disabled": True,
            "guard": "episode_role_only",
        }
        fd = os.open(receipt, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return connection
    except BaseException:
        connection.close()
        raise


def main():
    expected_role, receipt, *arguments = sys.argv[1:]
    if expected_role:
        import snowflake.connector

        original = snowflake.connector.connect

        def scoped_connect(*args, **kwargs):
            if kwargs.get("role") != expected_role:
                raise RuntimeError("dbt attempted a connection outside its assigned role")
            return guard_connection(original(*args, **kwargs), expected_role, receipt)

        snowflake.connector.connect = scoped_connect
    from dbt.cli.main import cli

    sys.argv = ["dbt", *arguments]
    cli(prog_name="dbt")


if __name__ == "__main__":
    main()
