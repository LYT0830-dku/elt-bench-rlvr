"""Recovery retains immutable observations and never resets paid accounting."""

import asyncio
import json
from pathlib import Path

import pytest

from elt_rl import batch_recovery
from elt_rl import train as recipe
from elt_rl.artifacts import write_json
from elt_rl.errors import IntegrityError
from elt_rl.lineage import evidence_ref


@pytest.fixture
def partial_case(tmp_path):
    config = recipe.TrainConfig("fixture", str(tmp_path), eval_episodes=10, temperature=0.8)
    inputs = {
        "task": {"fixture": True},
        "implementation_sha256": {"fixture.py": "a"},
        "reward_version": "execution_reward_v1",
    }
    label, seed = "eval-batch-0000", 100000
    save = tmp_path / "save.json"
    write_json(
        save,
        {
            "status": "confirmed",
            "updates": 0,
            "state_result": {"path": "fixture://state/0"},
            "sampler_result": {"path": "fixture://sampler/0"},
        },
    )
    checkpoint = tmp_path / "checkpoint-0000.json"
    write_json(
        checkpoint,
        {
            "updates": 0,
            "resume_allowed": True,
            "path": "fixture://state/0",
            "sampler_path": "fixture://sampler/0",
            "save_receipt": evidence_ref(save),
            "lineage": {},
            "fingerprint": {"evaluation_protocol": {}},
        },
    )
    known = {
        "checkpoint": evidence_ref(checkpoint),
        "size": 10,
        "seed": seed,
        "sampler_path": "fixture://sampler/0",
    }
    for index in range(10):
        path = tmp_path / "episodes" / f"{label}-{index:03d}.json"
        root = path.with_name(path.stem + "-files") / "exploration"
        write_json(
            root / "runtime.json",
            {
                "events": [
                    {
                        "kind": "warehouse",
                        "state": "closed",
                        "remaining_schemas": [],
                        "schema_creation_pending": [],
                        "role_exists": False,
                        "role_creation_pending": False,
                    },
                    {"kind": "cleanup", "status": "closed", "failures": []},
                ]
            },
        )
        write_json(
            root / "sources/resources.json",
            {
                "backend": "native_services_v1",
                "state": "closed",
                "listeners": [],
                "processes": [{"pid": 1, "running": False}],
            },
        )
        complete = index not in {0, 8, 9}
        official = {"official_score": 3 / 7, "official_success": False}
        report = {
            **official,
            "training_reward": 3 / 7,
            "pipeline_valid": True,
            "execution_success": True,
            "success": False,
            "training_eligible": True,
            "reward_version": "execution_reward_v1",
        }
        write_json(
            path,
            {
                "format": "elt_episode_v4",
                "mode": "native",
                "runtime": {"fixture": True},
                "episode_id": path.stem,
                "fingerprint": inputs["task"],
                "implementation_sha256": inputs["implementation_sha256"],
                "destination": config.destination,
                "done": complete,
                "termination": "submitted" if complete else "infrastructure_CancelledError",
                "report": report if complete else None,
                "execution_result": {
                    "kind": "submission",
                    "execution_success": True,
                    "pipeline_valid": True,
                    "execution": {},
                    "official": official,
                },
                "turns": 1,
                "sampling": [
                    {
                        "status": "sampled",
                        "seed": seed + 1009 * index,
                        "temperature": 0.8,
                        "top_p": 1.0,
                        "top_k": -1,
                        "tokens": [1],
                        "logprobs": [-0.5],
                        "output_tokens": 1,
                        "sequence_id": "offline",
                    }
                ],
            },
        )
    args = dict(label=label, size=10, seed=seed, inputs=inputs, config=config)
    return tmp_path, known, args


def test_partial_keeps_seven_retries_only_original_three_and_preserves_budget(partial_case):
    out, known, args = partial_case
    frozen = {
        p.name: p.read_bytes()
        for p in (out / "episodes").glob("*.json")
        if p.stem[-3:] not in {"000", "008", "009"}
    }
    cached, journal = batch_recovery.prepare_partial(out, known, **args)
    assert sorted(cached) == list(range(1, 8))
    assert [i for i in range(10) if i not in cached] == [0, 8, 9]
    repeated, _ = batch_recovery.prepare_partial(out, known, **args)
    assert sorted(repeated) == sorted(cached)
    assert len(json.loads(Path(journal["artifact_path"]).read_text())["archives"]) == 6
    assert all((out / "episodes" / name).read_bytes() == data for name, data in frozen.items())


def test_recovery_finishes_interrupted_archive_move(partial_case, monkeypatch):
    out, known, args = partial_case
    original = Path.rename
    calls = 0

    def interrupt(path, target):
        nonlocal calls
        value = original(path, target)
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        return value

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(asyncio.CancelledError):
        batch_recovery.prepare_partial(out, known, **args)
    monkeypatch.setattr(Path, "rename", original)
    cached, _ = batch_recovery.prepare_partial(out, known, **args)
    assert sorted(cached) == list(range(1, 8))
    assert not (out / "episodes/eval-batch-0000-000.json").exists()


@pytest.mark.parametrize("mutation", ["seed", "temperature", "cleanup", "weights", "source"])
def test_recovery_rejects_changed_evidence(partial_case, mutation):
    out, known, args = partial_case
    path = out / "episodes/eval-batch-0000-001.json"
    raw = json.loads(path.read_text())
    if mutation in {"seed", "temperature"}:
        raw["sampling"][0][mutation] = 0
    elif mutation == "source":
        raw["implementation_sha256"] = {"fixture.py": "bad"}
    elif mutation == "weights":
        known["sampler_path"] = "fixture://different-weights"
    else:
        f = path.with_name(path.stem + "-files") / "exploration/sources/resources.json"
        value = json.loads(f.read_text())
        value["state"] = "running"
        write_json(f, value)
    write_json(path, raw)
    with pytest.raises(IntegrityError):
        batch_recovery.prepare_partial(out, known, **args)


def test_completed_artifact_mutation_after_recovery_is_rejected(partial_case):
    out, known, args = partial_case
    batch_recovery.prepare_partial(out, known, **args)
    path = out / "episodes/eval-batch-0000-002.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(IntegrityError, match="digest"):
        batch_recovery.prepare_partial(out, known, **args)


def test_partial_training_cannot_resample(partial_case):
    out, known, args = partial_case
    with pytest.raises(IntegrityError, match="training"):
        batch_recovery.prepare_partial(out, known, **{**args, "label": "train-batch-0001"})


def test_cancelled_after_grading_retains_actual_report_outside_cohort(partial_case):
    out, known, args = partial_case
    path = out / "episodes/eval-batch-0000-007.json"
    raw = json.loads(path.read_text())
    raw["termination"] = "infrastructure_CancelledError"
    write_json(path, raw)
    frozen = path.read_bytes()
    cached, journal = batch_recovery.prepare_partial(out, known, **args)
    assert 7 not in cached
    record = json.loads(Path(journal["artifact_path"]).read_text())
    archived = next(m for m in record["archives"] if m["source"] == str(path))
    assert Path(archived["target"]).read_bytes() == frozen
    assert json.loads(frozen)["report"]["training_reward"] == 3 / 7
