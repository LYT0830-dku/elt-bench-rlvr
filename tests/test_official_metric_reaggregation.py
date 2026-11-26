"""Correct a synthetic historical multi-target cohort without rewriting its source."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_historical_reaggregation_preserves_original_rewards_and_files(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "official_metric_reaggregation",
        Path(__file__).parents[1] / "tools/reaggregate_official_metrics.py",
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    run = tmp_path / "original"
    (run / "episodes").mkdir(parents=True)
    task = SimpleNamespace(
        id="fixture",
        targets=[
            SimpleNamespace(name="a", columns=("ID",)),
            SimpleNamespace(name="b", columns=("ID",)),
        ],
    )
    raw = {
        "episode_id": "train-batch-0001-000",
        "task": task.id,
        "done": True,
        "termination": "submitted",
        "report": {
            "official_srdel": True,
            "official_srdt": False,
            "official_success": False,
            "stage2_log": "Table: a\nMatched columns: ['ID']\nTable: b\nUnmatched columns: ['ID']\n",
        },
    }
    episode = run / "episodes/train-batch-0001-000.json"
    episode.write_text(json.dumps(raw))
    digest = hashlib.sha256(episode.read_bytes()).hexdigest()
    cohort = {
        "n": 1,
        "label": "train-batch-0001",
        "official_srdt": 0.0,
        "episodes": [
            {
                "episode_id": raw["episode_id"],
                "reward": 0.25,
                "artifact": {"artifact_path": str(episode), "sha256": digest},
            }
        ],
    }
    protocol = run / "protocol-state.json"
    protocol.write_text(json.dumps({"batches": [{"rollout": cohort}]}))
    originals = {p: p.read_bytes() for p in [episode, protocol]}
    result = tool.reaggregate(run, task, tmp_path / "corrected")
    corrected = json.loads((tmp_path / "corrected/rollouts.json").read_text())[0]
    assert result["changed_rollouts"] == 1 and result["reward_changed"] is False
    assert corrected["official_srdt"] == 0.5 and corrected["reward"] == 0.25
    assert corrected["end_to_end_success"] is False
    assert all(p.read_bytes() == data for p, data in originals.items())
