"""Free component contracts and one real multi-target native ELT execution."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from elt_rl.artifacts import valid_submission_receipt
from elt_rl.cli import experiment
from elt_rl.environment import ELTEnvironment
from elt_rl.errors import IntegrityError
from elt_rl.execution_scoring import verify_scoring_receipt
from elt_rl.model_registry import MODELS
from elt_rl.official_grader import parse_target_columns
from elt_rl.registry import UnsupportedCapability
from elt_rl.source_adapters import SOURCES, decode
from elt_rl.task_scheduling import TaskSchedule, expected_task_fingerprint
from elt_rl.task_spec import load_task_spec
from elt_rl.train import TrainConfig


@pytest.fixture
def multi_pack(tmp_path):
    """Composite date key, boolean, decimal, two unequal targets, real local file."""
    root = tmp_path / "pack"
    root.mkdir()
    project = root / "project"
    project.mkdir()
    (root / "runtime.json").write_text('{"execution_backend":"native"}')
    (root / "events.csv").write_text(
        "tenant,day,active,amount\na,2026-01-01,true,4\na,2026-01-01,true,6\na,2026-01-02,false,2\nb,2026-01-01,true,3\n"
    )
    (root / "daily.csv").write_text(
        "tenant,day,active,total\na,2026-01-01,True,10\na,2026-01-02,False,2\nb,2026-01-01,True,3\n"
    )
    (root / "tenants.csv").write_text("tenant,total\na,12\nb,3\n")
    manifest = {
        "name": "multi_fixture",
        "description": "Aggregate events by tenant/day and tenant.",
        "mode": "local_test",
        "project_dir": "project",
        "runtime": {"config_file": "runtime.json"},
        "sources": [
            {
                "name": "events",
                "kind": "local_file",
                "columns": ["tenant", "day", "active", "amount"],
                "options": {"path": "events.csv", "format": "csv"},
            }
        ],
        "targets": [
            {
                "name": "daily",
                "columns": ["tenant", "day", "active", "total"],
                "keys": ["tenant", "day"],
                "numeric": ["total"],
                "expected": "daily.csv",
            },
            {
                "name": "tenants",
                "columns": ["tenant", "total"],
                "keys": ["tenant"],
                "numeric": ["total"],
                "expected": "tenants.csv",
            },
        ],
        "reference_sha256": {
            name: hashlib.sha256((root / (name + ".csv")).read_bytes()).hexdigest()
            for name in ("daily", "tenants")
        },
    }
    (root / "task.yaml").write_text(yaml.safe_dump(manifest))
    specification = {
        "format": "elt_task_spec_v1",
        "id": "multi_fixture",
        "manifest": "task.yaml",
        "evaluation": {
            "provider": "local_fixture",
            "warehouse_backends": ["duckdb"],
            "metrics": ["fixture_success"],
        },
        "alignment": {
            "type": "primary_key",
            "verified": True,
            "evidence": "Declared GROUP BY keys; reference uniqueness checked.",
        },
        "targets": [
            {
                "name": "daily",
                "keys": ["tenant", "day"],
                "fields": [
                    {"name": "tenant", "kind": "string", "role": "identifier"},
                    {"name": "day", "kind": "date", "role": "identifier"},
                    {"name": "active", "kind": "boolean"},
                    {"name": "total", "kind": "decimal"},
                ],
            },
            {
                "name": "tenants",
                "keys": ["tenant"],
                "fields": [
                    {"name": "tenant", "kind": "string", "role": "identifier"},
                    {"name": "total", "kind": "decimal"},
                ],
            },
        ],
    }
    path = root / "spec.yaml"
    path.write_text(yaml.safe_dump(specification))
    return path


def test_task_alignment_assets_and_private_public_contract(multi_pack):
    task = load_task_spec(multi_pack)
    assert len(task.reward_targets) == 2
    assert task.reward_targets[0].keys == ("tenant", "day")
    assert task.public() == task.execution_task.public()
    assert "reference" not in json.dumps(task.public()).lower()
    assert "framework_task_contract" in task.fingerprint()
    assert task.fingerprint() == json.loads(json.dumps(task.fingerprint()))
    with pytest.raises(UnsupportedCapability, match="alignment"):
        replace(task, alignment_verified=False).verify()
    Path(task.source_specs[0].configuration["path"]).write_text("modified")
    with pytest.raises(IntegrityError, match="digest"):
        task.verify()


def test_config_schedule_and_selected_resume_identity(multi_pack, tmp_path):
    raw = {
        "format": "elt_framework_v1",
        "run_id": "free-contract",
        "output": "out",
        "task": {"id": "multi_fixture", "spec": str(multi_pack)},
        "model": {"id": "Qwen/Qwen3-8B", "renderer_effort": None},
        "warehouse": {"backend": "duckdb"},
        "reward": {"type": "execution"},
        "training": {"batches": 2, "group_size": 3},
        "limits": {"max_tokens": 8192},
        "evaluation": {"interval": 1, "rollouts": 2, "final_rollouts": 2},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = experiment(path)
    assert config.model == "Qwen/Qwen3-8B" and config.max_tokens == 8192
    assert config.renderer == "qwen3" and config.batches == 2 and not config.allow_remote
    raw["training"]["new_algorithm"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="training parameter"):
        experiment(path)
    groups = [
        {"task_id": "a", "specification": "a.yaml"},
        {"task_id": "b", "specification": "b.yaml"},
    ]
    schedule = TaskSchedule(groups, "b")
    assert [schedule.select(f"train-batch-{i:04d}").task_id for i in range(1, 5)] == [
        "a",
        "b",
        "a",
        "b",
    ]
    inputs = {
        "task": {
            "task_schedule": groups,
            "evaluation_task": "b",
            "tasks": {"a": {"id": 1}, "b": {"id": 2}},
        }
    }
    assert expected_task_fingerprint(inputs, "eval-batch-0000") == {"id": 2}
    assert expected_task_fingerprint(inputs, "train-batch-0001") == {"id": 1}


def test_reward_reproduction_accepts_externally_verified_reference_assets(multi_pack, tmp_path):
    import importlib.util
    from dataclasses import asdict

    from elt_rl.execution_scoring import load_table

    spec = importlib.util.spec_from_file_location(
        "saved_reward_verifier", Path(__file__).parents[1] / "tools/verify_saved_rewards.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    task = load_task_spec(multi_pack)
    declared = {t.name: t for t in task.targets}
    tables = {
        t.name: asdict(load_table(declared[t.name].expected, t, max_rows=task.max_reference_rows))
        for t in task.reward_targets
    }
    data = {
        "task_id": task.id,
        "targets": [asdict(t) for t in task.reward_targets],
        "configuration": {"type": "pure_cell", "version": "pure_cell_accuracy_v1"},
        "episodes": [
            {
                "episode": "fixture",
                "predictions": tables,
                "stage1_passed": True,
                "submission_valid": True,
                "reward": 1.0,
                "official_columns": {
                    t.name: dict.fromkeys([f.name for f in t.fields], True)
                    for t in task.reward_targets
                },
            }
        ],
    }
    path = tmp_path / "public-prediction.json"
    path.write_text(json.dumps(data, default=str))
    result = verifier.verify(path, reference_task_spec=multi_pack)
    assert result["status"] == "passed" and result["maximum_absolute_difference"] == 0
    data["task_id"] = "another_task"
    path.write_text(json.dumps(data, default=str))
    with pytest.raises(ValueError, match="differs"):
        verifier.verify(path, reference_task_spec=multi_pack)


@pytest.mark.parametrize("enabled,expected_total", [(True, 218), (False, 128)])
def test_cli_plan_counts_only_enabled_evaluations(monkeypatch, capsys, enabled, expected_total):
    from elt_rl import cli

    config = SimpleNamespace(
        validate=lambda: None,
        run_id="offline-plan",
        model="registered-fixture-model",
        batches=16,
        group_size=8,
        evaluation_interval=4,
        eval_episodes=10,
        final_eval_episodes=20,
        evaluation_enabled=enabled,
    )
    monkeypatch.setattr(cli, "experiment", lambda path: config)
    cli.main(["inspect-experiment", "--config", "offline.yaml"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["total_rollouts"] == expected_total
    assert plan["process_evaluation_rollouts"] == (50 if enabled else 0)
    assert plan["final_evaluation_rollouts"] == (40 if enabled else 0)


def test_model_and_checkpoint_capabilities_fail_before_service():
    inkling, qwen = MODELS.get("thinkingmachines/Inkling-2-Mini"), MODELS.get("Qwen/Qwen3-8B")
    assert type(inkling.codec) is not type(qwen.codec)
    qwen.validate(renderer="qwen3", effort=None, training=True, context_tokens=32768)
    with pytest.raises(UnsupportedCapability, match="renderer"):
        qwen.validate(renderer="tml_v0")
    with pytest.raises(UnsupportedCapability, match="effort"):
        qwen.validate(effort=0)
    with pytest.raises(UnsupportedCapability, match="context"):
        qwen.validate(context_tokens=64000)
    with pytest.raises(UnsupportedCapability, match="another model"):
        qwen.validate_checkpoint({"config": {"model": inkling.id}})
    with pytest.raises(UnsupportedCapability, match="contract changed"):
        qwen.validate_checkpoint({"config": {"model": qwen.id}, "model_adapter": {}})
    qwen.validate_checkpoint({"config": {"model": qwen.id}, "model_adapter": qwen.fingerprint()})


def test_source_capabilities_and_decode():
    assert decode(b'[{"ID":1,"value":"a"}]', "json", ["id"]) == [{"id": 1}]
    assert decode(b"id,value\n2,b\n", "csv", ["value"]) == [{"value": "b"}]
    with pytest.raises(UnsupportedCapability, match="MongoDB"):
        SOURCES.validate([SimpleNamespace(kind="mongodb")])
    with pytest.raises(UnsupportedCapability):
        SOURCES.get("unknown")


def test_qwen_official_cookbook_tokenizer_tool_roundtrip():
    """Actual tokenizer and renderer, with local fixture tokens and no API client."""
    model = MODELS.get("Qwen/Qwen3-8B")
    _, renderer = model.renderer()
    observation = {
        "instructions": "Use structured tools.",
        "execution_backend": "native",
        "task": {"name": "transport"},
    }
    initial = model.codec.initial_messages(observation, renderer)
    prompt = renderer.tokenizer.decode(renderer.build_generation_prompt(initial).to_ints())
    assert "<tools>" in prompt and "write_file" in prompt
    arguments = {"path": "dbt/models/x.sql", "content": "select '汉字' as value"}
    wire = (
        "<tool_call>\n"
        + json.dumps({"name": "write_file", "arguments": arguments}, ensure_ascii=False)
        + "\n</tool_call><|im_end|>"
    )
    tokens = renderer.tokenizer.encode(wire, add_special_tokens=False)
    message, termination = renderer.parse_response(tokens)
    assert model.codec.parse_action(message, termination, "") == {"tool": "write_file", **arguments}
    observations = model.codec.observations(message, {"written": True})
    renderer.build_generation_prompt(initial + [message] + observations)
    assert observations[0]["role"] == "tool"


def test_official_target_identity_preserves_shared_column_names():
    targets = [
        SimpleNamespace(name="a", columns=("id", "value")),
        SimpleNamespace(name="b", columns=("id", "other")),
    ]
    log = "Table: a\nMatched columns: ['ID', 'VALUE']\nTable: b\nMatched columns: ['ID']\nUnmatched columns: ['OTHER']\n"
    result = parse_target_columns(log, targets, stage1_passed=True)
    assert result["a"]["matched"] == ["ID", "VALUE"] and result["b"]["matched"] == ["ID"]
    with pytest.raises(Exception, match="target identity"):
        parse_target_columns("Matched columns: ['ID']", targets, stage1_passed=True)


def test_real_multi_target_native_elt_private_reward_and_recovery(multi_pack):
    task = load_task_spec(multi_pack)
    reward = {"type": "pure_cell", "version": "pure_cell_accuracy_v1"}
    config = {
        "format": "native_loading_v1",
        "destination": {"database": "assigned", "schema": "airbyte_schema"},
        "connections": [
            {
                "name": "events",
                "source": "events",
                "connector": "local_file",
                "format": "csv",
                "table": "events",
                "columns": {
                    "tenant": "VARCHAR",
                    "day": "DATE",
                    "active": "BOOLEAN",
                    "amount": "DOUBLE",
                },
                "sync_mode": "full_refresh_overwrite",
            }
        ],
    }
    files = {
        "elt/loading.json": json.dumps(config),
        "dbt/dbt_project.yml": "name: audit\nversion: '1.0'\nconfig-version: 2\nprofile: elt\nmodel-paths: [models]\nmodels:\n  audit:\n    +materialized: table\n",
        "dbt/models/sources.yml": "version: 2\nsources:\n  - name: raw\n    schema: airbyte_schema\n    tables:\n      - name: events\n",
        "dbt/models/daily.sql": "select tenant, day, max(active) as active, sum(amount) as total from {{ source('raw', 'events') }} group by tenant, day",
        "dbt/models/tenants.sql": "select tenant, sum(amount) as total from {{ source('raw', 'events') }} group by tenant",
    }
    with ELTEnvironment(task, reward_config=reward, run_id="free", model_id="test") as env:
        env.initial()
        for path, content in files.items():
            assert (
                "error"
                not in env.step(
                    {"tool": "write_file", "path": path, "content": content}
                ).observation
            )
        assert "error" not in env.step({"tool": "loading", "command": "apply"}).observation
        job = env.step({"tool": "sync", "connection": "native_connection.events"}).observation[
            "job_id"
        ]
        assert "error" not in env.step({"tool": "wait_sync", "job_id": job}).observation
        ran = env.step({"tool": "dbt", "command": "run"})
        assert ran.observation.get("exit_code") == 0, ran.observation
        result = env.step({"tool": "submit"})
        assert result.done and result.reward == 1, env.report
        assert not any(
            key in result.observation
            for key in ("column_score", "reward", "targets", "matched", "reward_result")
        )
    raw = env.artifact()
    assert valid_submission_receipt(raw)
    assert verify_scoring_receipt(raw, reward)["reward"] == 1
    assert verify_scoring_receipt(raw, reward)["reward"] == 1  # Idempotent recovery, no calls.
    with pytest.raises(Exception, match="frozen selected reward"):
        verify_scoring_receipt(raw, {"type": "unsupported_mode"})
    saved = raw["report"]["reward_result"]["artifact"]["artifact_path"]
    scored = json.loads(Path(saved).read_text())
    prediction = next(iter(scored["targets_snapshot"].values()))["prediction"]["artifact_path"]
    with open(prediction, "a", newline="") as stream:
        csv.writer(stream).writerow(["tampered"])
    with pytest.raises(Exception, match="fingerprint changed"):
        verify_scoring_receipt(raw, reward)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_tokens": -1},
        {"max_turns": True},
        {"max_seconds": float("nan")},
        {"context_tokens": 100},
    ],
)
def test_invalid_eval_limits_are_free_failures(limits):
    config = TrainConfig("a", "b", evaluation_limits=limits)
    with pytest.raises(ValueError):
        config.validate()
