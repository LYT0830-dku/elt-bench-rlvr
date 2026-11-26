"""Native configuration remains declarative and loading obeys named column mapping."""

import json

import pytest

from elt_rl.environment import ELTEnvironment


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://169.254.169.254/"),
        ("private_key_file", "/home/another-user/key"),
        ("python", "open('/etc/passwd').read()"),
        ("sql", "select * from another_schema.secret"),
    ],
)
def test_loading_rejects_external_capabilities(task, pipeline_path, field, value):
    original = json.loads((pipeline_path.parent / "elt/loading.json").read_text())
    original["connections"][0][field] = value
    with ELTEnvironment(task) as env:
        env.step(
            {"tool": "write_file", "path": "elt/loading.json", "content": json.dumps(original)}
        )
        feedback = env.step({"tool": "loading", "command": "apply"})
        assert "error" in feedback.observation
        assert not env.runtime.applied and not env.runtime.jobs


def test_apply_does_not_load_and_append_maps_column_names(task, pipeline_path):
    config = json.loads((pipeline_path.parent / "elt/loading.json").read_text())
    connection = config["connections"][0]
    assert list(connection["columns"]) == ["customer_id", "customer_name"]
    config["connections"] = [connection]
    with ELTEnvironment(task) as env:

        def apply_sync():
            env.step(
                {"tool": "write_file", "path": "elt/loading.json", "content": json.dumps(config)}
            )
            assert env.step({"tool": "loading", "command": "apply"}).observation["applied"]
            address = "native_connection." + connection["name"]
            job = env.step({"tool": "sync", "connection": address}).observation["job_id"]
            assert (
                env.step({"tool": "wait_sync", "job_id": job}).observation["state"] == "succeeded"
            )

        env.step({"tool": "write_file", "path": "elt/loading.json", "content": json.dumps(config)})
        env.step({"tool": "loading", "command": "apply"})
        result = env.step({"tool": "query", "sql": "SELECT * FROM airbyte_schema.customers"})
        assert "error" in result.observation and not env.runtime.jobs
        apply_sync()
        initial = env.runtime.warehouse.output("customers")
        connection["columns"] = dict(reversed(list(connection["columns"].items())))
        connection["sync_mode"] = "full_refresh_append"
        apply_sync()
        actual = env.runtime.warehouse.output("customers")
        assert actual.columns == initial.columns
        assert sorted(actual.rows) == sorted(initial.rows * 2)
