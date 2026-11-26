"""Saved-table rescoring never invents missing historical predictions."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from elt_rl.artifacts import write_json
from elt_rl.lineage import evidence_ref
from elt_rl.rewards import RewardField, RewardTarget


def test_offline_actual_snapshot_and_missing_history_are_separate(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "offline_rescoring", Path(__file__).parents[1] / "tools/rescore_saved_rollouts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    (run / "episodes").mkdir(parents=True)
    reference, prediction = tmp_path / "reference.csv", tmp_path / "prediction.csv"
    reference.write_text("id,value\n1,yes\n2,yes\n")
    prediction.write_text("id,value\n1,yes\n")
    scoring = tmp_path / "original-scoring.json"
    write_json(
        scoring,
        {
            "targets_snapshot": {
                "unseen_target": {
                    "reference": evidence_ref(reference),
                    "prediction": evidence_ref(prediction),
                }
            }
        },
    )
    for index in range(2):
        write_json(
            run / "episodes" / f"eval-batch-0000-{index:03d}.json",
            {
                "episode_id": f"eval-batch-0000-{index:03d}",
                "done": True,
                "termination": "submitted",
                "report": {
                    "pipeline_valid": True,
                    "execution_success": True,
                    "training_reward": 0.123,
                    **(
                        {"reward_result": {"artifact": evidence_ref(scoring)}} if index == 0 else {}
                    ),
                },
            },
        )
    write_json(run / "protocol-state.json", {"batches": []})
    # A completed grader result remains usable for offline diagnosis even if
    # its async delivery was cancelled. It does not become a selected cohort.
    cancelled = json.loads((run / "episodes/eval-batch-0000-000.json").read_text())
    cancelled["episode_id"] = "eval-batch-0000-002"
    cancelled["termination"] = "infrastructure_CancelledError"
    receipt = {**cancelled["report"]["reward_result"], "reward": 0.123, "config_sha256": "a" * 64}
    official = {"official_score": 0.123, "official_success": False}
    cancelled["report"].update(**official, success=False, reward_result=receipt)
    cancelled["execution_result"] = {
        "kind": "submission",
        "execution_success": True,
        "pipeline_valid": True,
        "execution": {},
        "official": official,
        "reward_result": receipt,
    }
    write_json(run / "episodes/eval-batch-0000-002.json", cancelled)
    target = RewardTarget(
        "unseen_target", (RewardField("id", "integer", "identifier"), RewardField("value")), ("id",)
    )
    task = SimpleNamespace(
        id="unseen_task",
        execution_task=SimpleNamespace(name="unseen_task"),
        verify=lambda: None,
        reward_targets=(target,),
        max_reference_rows=100,
    )
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*.json")}
    result = module.rescore_run(run, task, tmp_path / "rescored")
    assert result["rescored"] == 2 and result["unavailable"] == 1
    assert result["episodes"][0]["cell_accuracy"] == 0.5
    assert result["episodes"][1]["cell_accuracy"] is None
    assert result["episodes"][2]["cell_accuracy"] == 0.5
    assert result["episodes"][2]["delivery_cancelled_after_grading"] is True
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    saved = json.loads((tmp_path / "rescored/rescores.json").read_text())
    assert saved["offline_only"] and saved["cohorts"][0]["complete_original_cohort"] is False
    prediction.write_text("id,value\n1,tampered\n")
    with pytest.raises(ValueError, match="digest differs"):
        module.rescore_run(run, task, tmp_path / "tampered")
