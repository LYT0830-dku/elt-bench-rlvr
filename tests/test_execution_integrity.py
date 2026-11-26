"""Security regressions with real local execution and explicit offline failure injection."""

import json
from types import SimpleNamespace

import pytest
from test_environment import execute

from elt_rl.environment import ELTEnvironment
from elt_rl.errors import InfrastructureError
from elt_rl.failures import check_dbt_result
from elt_rl.pipeline import load_pipeline


def test_submission_rejects_unexecuted_file_and_requires_new_run(task, pipeline_path):
    with ELTEnvironment(task) as env:
        execute(env, load_pipeline(pipeline_path), submit=False)
        path = "dbt/models/customer_sales.sql"
        old = env.project.read(path)
        changed = old.replace("SUM(orders.amount)", "SUM(ABS(orders.amount))")
        assert changed != old
        env.step({"tool": "write_file", "path": path, "content": changed})
        rejected = env.step({"tool": "submit"})
        assert not rejected.done and rejected.reward == 0 and "error" in rejected.observation
        assert not (env.root / "submission").exists()
        env.step({"tool": "dbt", "command": "run"})
        accepted = env.step({"tool": "submit"})
        assert accepted.done and accepted.reward == pytest.approx(15 / 16)


def test_resync_invalidates_transformation_and_obsolete_job(task, pipeline_path):
    with ELTEnvironment(task) as env:
        execute(env, load_pipeline(pipeline_path), submit=False)
        connection = env.runtime.recipe["sync_connections"][0]
        old_job = env.runtime.latest_jobs[connection]
        new_job = env.step({"tool": "sync", "connection": connection}).observation["job_id"]
        assert not env.runtime.valid()
        old = env.step({"tool": "wait_sync", "job_id": old_job})
        assert "obsolete" in old.observation["error"]
        env.step({"tool": "wait_sync", "job_id": new_job})
        assert not env.runtime.valid()
        env.step({"tool": "dbt", "command": "run"})
        assert env.step({"tool": "submit"}).reward == 1


def test_required_sources_and_loading_files_are_part_of_receipt(task, pipeline_path):
    with ELTEnvironment(task) as env:
        execute(env, load_pipeline(pipeline_path), submit=False)
        assert env.runtime.valid()
        address = env.runtime.recipe["sync_connections"].pop()
        env.runtime.synced.remove(address)
        assert not env.runtime.valid()
        env.runtime.synced.add(address)
        env.runtime.recipe["sync_connections"].append(address)
        text = env.project.read("elt/loading.json")
        env.project.write("elt/loading.json", text + "\n ")
        assert not env.runtime.valid()


@pytest.mark.parametrize(
    "message,policy",
    [
        ("Compilation Error in model x: bad column", True),
        ("SQL compilation error: bad table", True),
        ("Database Error: connection reset", False),
        ("Unknown worker failure", False),
    ],
)
def test_dbt_infrastructure_does_not_become_zero_reward(message, policy):
    result = {"exit_code": 1, "output": message}
    if policy:
        check_dbt_result(result)
    else:
        with pytest.raises(InfrastructureError):
            check_dbt_result(result)


@pytest.mark.parametrize(
    "kind,policy", [("config_error", True), ("system_error", False), ("transient_error", False)]
)
def test_native_loading_infrastructure_does_not_become_zero_reward(
    kind, policy, task, pipeline_path, monkeypatch
):
    from elt_rl.errors import ActionError

    with ELTEnvironment(task) as env:
        config = (pipeline_path.parent / "elt/loading.json").read_text()
        env.step({"tool": "write_file", "path": "elt/loading.json", "content": config})
        env.step({"tool": "loading", "command": "apply"})

        def fail(*args, **kwargs):
            if policy:
                raise ActionError("Invalid model-selected destination type")
            raise InfrastructureError("Unavailable destination connection")

        monkeypatch.setattr(env.runtime, "_write_table", fail)
        if policy:
            job = env.step(
                {"tool": "sync", "connection": "native_connection.customers"}
            ).observation["job_id"]
            result = env.step({"tool": "wait_sync", "job_id": job})
            assert not result.done and result.reward == 0
        else:
            with pytest.raises(InfrastructureError):
                job = env.step(
                    {"tool": "sync", "connection": "native_connection.customers"}
                ).observation["job_id"]
                env.step({"tool": "wait_sync", "job_id": job})
        if policy:
            assert env.report is None
        else:
            assert env.report["training_reward"] is None
            assert env.report["failure_type"] == "infrastructure"
            assert env.report["training_eligible"] is False


def test_grader_swallowed_infrastructure_error_aborts(tmp_path, monkeypatch):
    from elt_rl.official_grader import OfficialGrader

    monkeypatch.setattr(
        "elt_rl.official_grader.task_material", lambda *a: ({}, {"revision": "fixture"})
    )
    root = tmp_path / "grade"

    def run(*args, **kwargs):
        evaluation = root / "evaluation"
        (evaluation / "agent_results/run/results.log").write_text("Success: cars schema verified\n")
        (evaluation / "agent_results/run/stage2.log").write_text("Error: connection was closed\n")
        (evaluation / "driver_errors.json").write_text(
            json.dumps(
                [
                    {
                        "operation": "fetchall",
                        "exception": "OperationalError",
                        "sqlstate": "08001",
                        "infrastructure": True,
                    }
                ]
            )
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("elt_rl.official_grader.subprocess.run", run)
    with pytest.raises(InfrastructureError, match="infrastructure failed"):
        OfficialGrader(tmp_path).grade(
            SimpleNamespace(
                dialect="snowflake", schema="RL_FIXTURE", database="FIXTURE", config={}
            ),
            SimpleNamespace(name="cars", targets=[]),
            root,
        )
