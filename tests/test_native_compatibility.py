"""Safe ordinary syntax and actionable execution feedback across task packs."""

import json
import threading

import pytest
import yaml

from elt_rl.capabilities import dbt_policy
from elt_rl.environment import ELTEnvironment
from elt_rl.errors import ActionError, InfrastructureError
from elt_rl.pipeline import load_pipeline, run_pipeline


@pytest.mark.parametrize("alias", ["STRING", "text", "varchar"])
def test_string_alias_and_plain_dbt_tags_execute_and_replay(task, pipeline_path, alias):
    pipeline = load_pipeline(pipeline_path)
    loading = json.loads(pipeline["files"]["elt/loading.json"])
    for entry in loading["connections"]:
        entry["columns"] = {
            name: alias if kind == "VARCHAR" else kind for name, kind in entry["columns"].items()
        }
    pipeline["files"]["elt/loading.json"] = json.dumps(loading)
    project = yaml.safe_load(pipeline["files"]["dbt/dbt_project.yml"])
    project["models"][project["name"]]["+tags"] = ["transfer_validation"]
    pipeline["files"]["dbt/dbt_project.yml"] = yaml.safe_dump(project)
    artifact = run_pipeline(task, pipeline)
    assert artifact["report"]["training_reward"] == 1
    assert artifact["report"]["correct_cells"] == 16
    assert artifact["runtime"]["pipeline_valid"]


def test_fast_failed_sync_reports_connection_column_and_value(task, pipeline_path, monkeypatch):
    loading = json.loads((pipeline_path.parent / "elt/loading.json").read_text())
    loading["connections"] = loading["connections"][:1]
    loading["connections"][0]["columns"]["customer_name"] = "DOUBLE"
    with ELTEnvironment(task) as env:
        env.step({"tool": "write_file", "path": "elt/loading.json", "content": json.dumps(loading)})
        env.step({"tool": "loading", "command": "apply"})
        submit = env.runtime.executor.submit

        def finish_before_dispatch_returns(*args):
            future = submit(*args)
            future.result(timeout=10)
            return future

        monkeypatch.setattr(env.runtime.executor, "submit", finish_before_dispatch_returns)
        result = env.step({"tool": "sync", "connection": "native_connection.customers"})
        assert result.observation["state"] == "failed"
        feedback = result.observation["feedback"]
        assert "source customers, column customer_name" in feedback
        assert "DOUBLE" in feedback and "source value" in feedback
        waited = env.step({"tool": "wait_sync", "job_id": result.observation["job_id"]})
        assert waited.observation["feedback"] == feedback
        rejected = env.step({"tool": "submit"})
        assert "native_connection.customers failed" in rejected.observation["error"]
        assert "No successful dbt run" in rejected.observation["error"]
        assert not rejected.done and env.report is None


def test_ignored_background_infrastructure_failure_cannot_be_policy_zero(
    task, pipeline_path, monkeypatch, tmp_path
):
    env = ELTEnvironment(task, artifact_path=tmp_path / "episode.json")
    release = threading.Event()
    try:
        loading = (pipeline_path.parent / "elt/loading.json").read_text()
        env.step({"tool": "write_file", "path": "elt/loading.json", "content": loading})
        env.step({"tool": "loading", "command": "apply"})

        def fail(*args):
            assert release.wait(10)
            raise InfrastructureError("Unavailable destination")

        monkeypatch.setattr(env.runtime, "_write_table", fail)
        result = env.step({"tool": "sync", "connection": "native_connection.customers"})
        job_id = result.observation["job_id"]
        release.set()
        # Only the host waits for the worker; the actor never calls wait_sync.
        with pytest.raises(InfrastructureError):
            env.runtime.futures[job_id].result(timeout=10)
        with pytest.raises(InfrastructureError, match="not inspected"):
            env.close()
        saved = json.loads((tmp_path / "episode.json").read_text())
        assert saved["report"]["training_reward"] is None
        assert saved["report"]["training_eligible"] is False
        assert saved["termination"] == "infrastructure_error"
        assert env.runtime.closed
    finally:
        release.set()
        env.close()


def test_plain_tags_do_not_allow_hooks_or_templates(tmp_path):
    root = tmp_path / "dbt"
    root.mkdir()
    project = {"name": "validation", "profile": "elt", "models": {"validation": {}}}
    config = project["models"]["validation"]
    for field, value in [
        ("+tags", {"run": "anything"}),
        ("+tags", ["{{ env_var('PRIVATE_KEY') }}"]),
        ("+pre-hook", "select external_function()"),
    ]:
        config.clear()
        config[field] = value
        (root / "dbt_project.yml").write_text(yaml.safe_dump(project))
        with pytest.raises(ActionError):
            dbt_policy(root, "snowflake")
