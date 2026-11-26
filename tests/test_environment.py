"""Project lifecycle regressions using real native extraction/loading, dbt and DuckDB."""

import concurrent.futures
import json
import time

import pytest

from elt_rl.environment import ELTEnvironment
from elt_rl.pipeline import load_pipeline, pipeline_actions, run_pipeline


def execute(env, pipeline, *, submit=True):
    env.initial()
    last_job = None
    for action in pipeline_actions(pipeline):
        if action["tool"] == "submit" and not submit:
            break
        if action.get("job_id") == "$last_job":
            action = {**action, "job_id": last_job}
        result = env.step(action)
        if action["tool"] == "sync":
            last_job = result.observation.get("job_id")
        assert "error" not in result.observation, result.observation
    return result


def test_end_to_end_submission_lifecycle(task, pipeline_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with ELTEnvironment(task, artifact_path="relative-result.json") as env:
        assert env.root.is_absolute()
        result = execute(env, load_pipeline(pipeline_path))
        assert result.done and result.reward == 1 and env.report["success"]
        assert env.artifact()["execution_result"]["execution_success"]
        assert (env.root / "submission/pipeline.json").is_file()
        with pytest.raises(RuntimeError):
            env.step({"tool": "submit"})
    assert env.artifact()["runtime"]["events"][-1]["status"] == "closed"


def test_failed_query_feedback_then_repair(task, pipeline_path):
    with ELTEnvironment(task) as env:
        result = env.step({"tool": "query", "sql": "SELECT x FROM airbyte_schema.nonexistent"})
        assert "error" in result.observation and not result.done and result.reward == 0
        result = execute(env, load_pipeline(pipeline_path))
        assert result.reward == 1


def test_empty_submission_cannot_pass(task):
    with ELTEnvironment(task) as env:
        result = env.step({"tool": "submit"})
        assert result.reward == 0 and not result.done
        assert "error" in result.observation and env.report is None


def test_wrong_answer_is_partial_not_success(task, pipeline_path):
    pipeline = load_pipeline(pipeline_path)
    original = dict(pipeline["files"])
    pipeline["files"] = {
        name: sql.replace("SUM(orders.amount)", "SUM(ABS(orders.amount))")
        for name, sql in pipeline["files"].items()
    }
    assert pipeline["files"] != original
    artifact = run_pipeline(task, pipeline)
    assert artifact["report"]["training_reward"] == pytest.approx(15 / 16)
    assert artifact["report"]["correct_cells"] == 15
    assert artifact["report"]["required_cells"] == 16
    assert not artifact["report"]["success"]


def test_episode_isolation_parallel(task, pipeline_path):
    pipeline = load_pipeline(pipeline_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_pipeline(task, pipeline), range(2)))
    assert all(r["report"]["training_reward"] == 1 for r in results)
    with ELTEnvironment(task) as fresh:
        result = fresh.step({"tool": "query", "sql": "SELECT * FROM airbyte_schema.customer_sales"})
        assert "error" in result.observation and result.reward == 0


@pytest.mark.parametrize(
    "action",
    ["not json", "[]", '{"tool":[]}', {"tool": "unknown"}],
)
def test_invalid_actions_return_feedback(task, action):
    with ELTEnvironment(task) as env:
        result = env.step(action)
        assert "error" in result.observation and result.reward == 0 and not result.done


def test_budget_terminations(task):
    with ELTEnvironment(task, max_turns=1) as env:
        result = env.step({"tool": "inspect_task"})
        assert result.done and result.reward == 0
    with ELTEnvironment(task, max_seconds=0.1) as env:
        env.started = time.monotonic() - 1
        result = env.step({"tool": "submit"})
        assert result.done and result.reward == 0


def test_no_private_task_content_in_prompt(task):
    with ELTEnvironment(task) as env:
        public = json.dumps(env.initial())
        assert "expected.csv" not in public and "/Users/" not in public
        assert "options" not in public


def test_sync_is_idempotent_and_failed_cast_preserves_previous_table(task, pipeline_path):
    config = json.loads((pipeline_path.parent / "elt/loading.json").read_text())
    connection = config["connections"][0]
    config["connections"] = [connection]
    with ELTEnvironment(task) as env:

        def apply_sync():
            env.step(
                {"tool": "write_file", "path": "elt/loading.json", "content": json.dumps(config)}
            )
            assert env.step({"tool": "loading", "command": "apply"}).observation["applied"]
            job = env.step(
                {"tool": "sync", "connection": "native_connection.customers"}
            ).observation["job_id"]
            return env.step({"tool": "wait_sync", "job_id": job})

        for _ in range(2):
            assert apply_sync().observation["state"] == "succeeded"
            assert env.runtime.warehouse._query(
                "SELECT COUNT(*) FROM airbyte_schema.customers", 1
            ).rows == ((4,),)
        connection["columns"]["customer_name"] = "BIGINT"
        result = apply_sync()
        assert result.observation["state"] == "failed"
        assert env.runtime.warehouse._query(
            "SELECT COUNT(*) FROM airbyte_schema.customers", 1
        ).rows == ((4,),)


def test_submission_scores_current_warehouse(task, pipeline_path):
    from elt_rl.capabilities import rendered_sql

    good = load_pipeline(pipeline_path)
    bad = load_pipeline(pipeline_path)
    name = "dbt/models/customer_sales.sql"
    bad["files"][name] = bad["files"][name].replace("SUM(orders.amount)", "SUM(ABS(orders.amount))")
    assert bad["files"][name] != good["files"][name]
    with ELTEnvironment(task) as env:
        execute(env, bad, submit=False)
        env.runtime.warehouse._query(
            "CREATE OR REPLACE TABLE airbyte_schema.customer_sales AS "
            + rendered_sql(good["files"][name], "duckdb"),
            0,
        )
        result = env.step({"tool": "submit"})
        assert result.reward == 1 and env.report["success"]


def test_unknown_task_settings_are_rejected(task_path, tmp_path):
    import yaml

    from elt_rl.task import Task

    data = yaml.safe_load(task_path.read_text())
    data["unexpected_setting"] = True
    path = tmp_path / "invalid-task.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Unknown task settings: unexpected_setting"):
        Task.load(path)
