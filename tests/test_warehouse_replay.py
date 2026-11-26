"""Real native dbt/DuckDB regressions: exploratory state must reproduce its bundle."""

import json

from test_environment import execute

from elt_rl.environment import ELTEnvironment
from elt_rl.pipeline import load_pipeline, run_pipeline


def test_removed_dbt_target_cannot_earn_reward_from_obsolete_table(task, pipeline_path):
    with ELTEnvironment(task) as env:
        execute(env, load_pipeline(pipeline_path), submit=False)
        env.step({"tool": "delete_file", "path": "dbt/models/customer_sales.sql"})
        env.step({"tool": "write_file", "path": "dbt/models/other.sql", "content": "select 1 as x"})
        assert env.step({"tool": "dbt", "command": "run"}).observation["exit_code"] == 0
        absent = env.step({"tool": "query", "sql": "select * from airbyte_schema.customer_sales"})
        assert "error" in absent.observation
        current = env.step({"tool": "submit"})
        replay = run_pipeline(task, load_pipeline(env.root / "submission/pipeline.json"))
        assert current.reward == replay["report"]["training_reward"] == 0


def test_removed_loading_connection_table_is_pruned(task, pipeline_path):
    with ELTEnvironment(task) as env:
        pipeline = load_pipeline(pipeline_path)
        execute(env, pipeline, submit=False)
        config = json.loads(env.project.read("elt/loading.json"))
        # A declared temporary source copy is loaded, then removed from the final config.
        extra = dict(config["connections"][0], name="abandoned", table="abandoned")
        config["connections"].append(extra)
        env.project.write("elt/loading.json", json.dumps(config))
        env.step({"tool": "loading", "command": "apply"})
        job = env.step({"tool": "sync", "connection": "native_connection.abandoned"}).observation[
            "job_id"
        ]
        env.step({"tool": "wait_sync", "job_id": job})
        assert env.runtime.warehouse.output("abandoned").rows
        config["connections"].pop()
        env.project.write("elt/loading.json", json.dumps(config))
        env.step({"tool": "loading", "command": "apply"})
        for entry in config["connections"]:
            job = env.step(
                {"tool": "sync", "connection": "native_connection." + entry["name"]}
            ).observation["job_id"]
            env.step({"tool": "wait_sync", "job_id": job})
        assert env.step({"tool": "dbt", "command": "run"}).observation["exit_code"] == 0
        assert (
            "error"
            in env.step(
                {"tool": "query", "sql": "select * from airbyte_schema.abandoned"}
            ).observation
        )
        assert env.step({"tool": "submit"}).reward == 1


def test_repeated_append_recipe_replays_exact_multiplicity(task, pipeline_path):
    pipeline = load_pipeline(pipeline_path)
    config = json.loads(pipeline["files"]["elt/loading.json"])
    config["connections"][0]["sync_mode"] = "full_refresh_append"
    pipeline["files"]["elt/loading.json"] = json.dumps(config)
    with ELTEnvironment(task) as env:
        execute(env, pipeline, submit=False)
        address = "native_connection." + config["connections"][0]["name"]
        before = env.runtime.warehouse.output(config["connections"][0]["table"])
        job = env.step({"tool": "sync", "connection": address}).observation["job_id"]
        env.step({"tool": "wait_sync", "job_id": job})
        assert len(env.runtime.warehouse.output(config["connections"][0]["table"]).rows) == 2 * len(
            before.rows
        )
        assert env.runtime.recipe["sync_connections"].count(address) == 2
        env.step({"tool": "dbt", "command": "run"})
        current = env.step({"tool": "submit"})
        replay = run_pipeline(task, load_pipeline(env.root / "submission/pipeline.json"))
        assert replay["report"]["training_reward"] == current.reward
        assert replay["runtime"]["recipe"]["sync_connections"].count(address) == 2


def test_append_inherits_old_configuration_blocks_submission(task, pipeline_path):
    with ELTEnvironment(task) as env:
        execute(env, load_pipeline(pipeline_path), submit=False)
        config = json.loads(env.project.read("elt/loading.json"))
        config["connections"][0]["sync_mode"] = "full_refresh_append"
        env.project.write("elt/loading.json", json.dumps(config))
        env.step({"tool": "loading", "command": "apply"})
        for entry in config["connections"]:
            job = env.step(
                {"tool": "sync", "connection": "native_connection." + entry["name"]}
            ).observation["job_id"]
            env.step({"tool": "wait_sync", "job_id": job})
        run = env.step({"tool": "dbt", "command": "run"})
        assert "older loading configuration" in run.observation["error"]
        assert not env.runtime.valid()
        assert not env.step({"tool": "submit"}).done


def test_append_cannot_inherit_an_abandoned_dbt_relation(task, pipeline_path):
    pipeline = load_pipeline(pipeline_path)
    pipeline["files"]["dbt/models/abandoned.sql"] = 'select * from {{ source("raw", "customers") }}'
    # Match the fixture's actual source name, rather than a hand-built warehouse table.
    source_yaml = pipeline["files"]["dbt/models/sources.yml"]
    import yaml

    source_name = yaml.safe_load(source_yaml)["sources"][0]["name"]
    pipeline["files"]["dbt/models/abandoned.sql"] = (
        "select * from {{ source('" + source_name + "', 'customers') }}"
    )
    with ELTEnvironment(task) as env:
        execute(env, pipeline, submit=False)
        env.project.delete("dbt/models/abandoned.sql")
        config = json.loads(env.project.read("elt/loading.json"))
        config["connections"][0]["table"] = "abandoned"
        config["connections"][0]["sync_mode"] = "full_refresh_append"
        env.project.write("elt/loading.json", json.dumps(config))
        env.step({"tool": "loading", "command": "apply"})
        for entry in config["connections"]:
            job = env.step(
                {"tool": "sync", "connection": "native_connection." + entry["name"]}
            ).observation["job_id"]
            env.step({"tool": "wait_sync", "job_id": job})
        assert "abandoned" in env.runtime.unreproducible_tables
        assert (
            "older loading configuration"
            in env.step({"tool": "dbt", "command": "run"}).observation["error"]
        )
        assert not env.step({"tool": "submit"}).done
