"""An infrastructure retry cannot discard graded samples or duplicate an update."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from elt_rl.artifacts import write_json
from elt_rl.errors import IntegrityError
from elt_rl.lineage import evidence_ref


@pytest.fixture
def retry_case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "training_retry", Path(__file__).parents[1] / "tools/retry_failed_training_cohort.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from elt_rl import train

    inputs = {"task": {"mode": "native"}, "implementation_sha256": "a" * 64}
    config = SimpleNamespace(group_size=2, seed=0)
    monkeypatch.setattr(train, "TrainConfig", lambda **_: config)
    monkeypatch.setattr(train, "preflight", lambda _: None)
    monkeypatch.setattr(train, "fingerprint", lambda *_: inputs)
    monkeypatch.setattr(
        train, "checked_checkpoint", lambda *_: {"updates": 0, "sampler_path": "base"}
    )
    run = tmp_path / "run"
    (run / "episodes").mkdir(parents=True)
    checkpoint = run / "checkpoint-0000.json"
    write_json(checkpoint, {"updates": 0})
    known = {
        "status": "dispatching",
        "size": 2,
        "seed": 1_000_000,
        "sampler_path": "base",
        "checkpoint": evidence_ref(checkpoint),
    }
    state = {
        "status": "failed",
        "exception": "InfrastructureError",
        "error": "dbt connection failed",
        "phase": "train-batch-0001",
        "completed_batches": 0,
        "attempted_batches": 1,
        "updates": 0,
        "config": {},
        "fingerprint": inputs,
        "current_checkpoint": evidence_ref(checkpoint),
        "cohorts": {"train-batch-0001": known},
        "sampling": {
            "input_tokens": 1234,
            "output_tokens": 42,
            "reserved_or_uncertain_tokens": 100,
        },
    }
    write_json(run / "protocol-state.json", state)
    for i in range(2):
        episode = run / "episodes" / f"train-batch-0001-{i:03d}.json"
        write_json(
            episode,
            {
                "format": "elt_episode_v4",
                "episode_id": episode.stem,
                "fingerprint": inputs["task"],
                "implementation_sha256": inputs["implementation_sha256"],
                "policy_checkpoint": known["checkpoint"],
                "report": None,
                "termination": "infrastructure_InfrastructureError"
                if i == 0
                else "infrastructure_CancelledError",
                "execution_result": None,
                "runtime": {"allocated": True},
            },
        )
        resources = episode.with_name(episode.stem + "-files") / "exploration"
        resources.mkdir(parents=True)
        write_json(
            resources / "runtime.json",
            {
                "events": [
                    {
                        "kind": "warehouse",
                        "backend": "duckdb",
                        "state": "closed",
                        "active_connections": 0,
                    },
                    {"kind": "cleanup", "status": "closed", "failures": []},
                ]
            },
        )
    return module, run, state


def test_retry_preserves_counters_archives_attempt_and_is_idempotent(retry_case):
    module, run, before = retry_case
    record = module.prepare_retry(run)
    after = json.loads((run / "protocol-state.json").read_text())
    assert record["status"] == "ready" and record["optimizer_dispatched"] is False
    assert after["sampling"] == before["sampling"]
    assert after["updates"] == 0 and after["completed_batches"] == 0
    assert after["cohorts"]["train-batch-0001"]["status"] == "retry_prepared"
    assert not list((run / "episodes").glob("*.json"))
    assert len(list((run / "infrastructure-training-attempts").rglob("train-batch-*.json"))) == 2
    original = (run / "protocol-state.json").read_bytes()
    assert module.prepare_retry(run) == record
    assert (run / "protocol-state.json").read_bytes() == original
    after["cohorts"]["train-batch-0001"]["status"] = "dispatching"
    write_json(run / "protocol-state.json", after)
    with pytest.raises(IntegrityError, match="already consumed"):
        module.prepare_retry(run)


@pytest.mark.parametrize("violation", ["graded", "optimizer", "fingerprint", "cleanup"])
def test_retry_rejects_unsafe_or_reward_bearing_group(retry_case, violation):
    module, run, _ = retry_case
    episode = run / "episodes/train-batch-0001-000.json"
    raw = json.loads(episode.read_text())
    if violation == "graded":
        raw["report"] = {"training_eligible": True, "training_reward": 0.5}
        write_json(episode, raw)
    elif violation == "optimizer":
        write_json(run / "optimization-batch-0001.json", {"status": "optimizer_dispatching"})
    elif violation == "fingerprint":
        raw["implementation_sha256"] = "b" * 64
        write_json(episode, raw)
    else:
        runtime = episode.with_name(episode.stem + "-files") / "exploration/runtime.json"
        write_json(runtime, {"events": [{"kind": "warehouse", "state": "allocated"}]})
    with pytest.raises(IntegrityError):
        module.prepare_retry(run)
    assert episode.exists()


def test_retry_finishes_interrupted_archive_without_another_retry(retry_case, monkeypatch):
    module, run, _ = retry_case
    original = module._finish_moves
    calls = []

    def interrupted(moves):
        original(moves[:1])
        calls.append(True)
        raise InterruptedError("simulated process death")

    monkeypatch.setattr(module, "_finish_moves", interrupted)
    with pytest.raises(InterruptedError):
        module.prepare_retry(run)
    monkeypatch.setattr(module, "_finish_moves", original)
    result = module.prepare_retry(run)
    assert calls == [True] and result["retry_index"] == 1
    assert result["status"] == "ready"
