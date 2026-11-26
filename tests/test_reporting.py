"""Report proof with synthetic offline records: include declines and no eval gradients."""

import csv
import json

import pytest

from elt_rl.artifacts import write_json
from elt_rl.reporting import render_run_report


def test_report_renders_all_figures_and_keeps_decline_and_eval_separation(tmp_path):
    out = tmp_path / "run"
    (out / "episodes").mkdir(parents=True)
    for label, reward in [
        ("train-batch-0001-000", 1 / 7),
        ("train-batch-0001-001", 5 / 7),
        ("eval-batch-0000-000", 3 / 7),
        ("final-base-000", 3 / 7),
        ("final-final-000", 2 / 7),
    ]:
        write_json(
            out / "episodes" / (label + ".json"),
            {
                "episode_id": label,
                "done": True,
                "termination": "submitted",
                "wall_clock_seconds": 2,
                "report": {
                    "training_reward": reward,
                    "official_srdel": True,
                    "official_srdt": False,
                    "matched_target_columns": round(reward * 7),
                    "success": False,
                    "official_score": reward,
                },
                "sampling": [
                    {
                        "status": "sampled",
                        "seed": 200000000,
                        "input_tokens": 100,
                        "output_tokens": 10,
                    }
                ],
            },
        )

    def cohort(reward):
        return {
            "n": 1,
            "successes": 0,
            "mean_reward": reward,
            "reward_std": 0,
            "official_srdel": 1,
            "official_srdt": 0,
            "end_to_end_success_rate": 0,
            "matched_columns": [round(reward * 7)],
            "episode_seed_set": [200000000],
            "rewards": [reward],
        }

    state = {
        "config": {},
        "status": "complete",
        "attempted_batches": 1,
        "completed_batches": 1,
        "updates": 1,
        "batches": [
            {
                "batch": 1,
                "updates": 1,
                "updates_before": 0,
                "rewards": [1 / 7, 5 / 7],
                "advantages": [-2 / 7, 2 / 7],
                "mean_reward": 3 / 7,
                "status": "effective_update_confirmed",
                "forward_backward_result": {"metrics": {"loss:sum": 0.1}},
            }
        ],
        "process_evaluations": [{"batch": 0, **cohort(3 / 7)}],
        "final_evaluations": {"base": cohort(3 / 7), "final": cohort(2 / 7)},
    }
    report = render_run_report(out, state)
    assert report["final_minus_base_mean_reward"] == pytest.approx(-1 / 7)
    assert "decreased" in (out / "report/REPORT.md").read_text()
    assert len(json.loads((out / "report/figure-index.json").read_text())["generated"]) == 9
    rows = list(csv.DictReader((out / "report/rollouts.csv").open()))
    for row in rows:
        if not row["episode_id"].startswith("train-"):
            assert row["advantage"] == "" and row["optimizer_applied"] == "False"


def test_pure_cell_report_exports_full_evaluation_statistics_and_cell_rates(tmp_path):
    from elt_rl.metrics import cell_statistics

    out = tmp_path / "pure"
    (out / "episodes").mkdir(parents=True)
    finals = {}
    for policy, counts in [("base", [2, 4]), ("final", [1, 2])]:
        results = []
        for index, count in enumerate(counts):
            mask = [
                {"target": "generic", "key": [str(j)], "column": "value", "correct": j < count}
                for j in range(4)
            ]
            result = {"cell_accuracy": count / 4, "cell_results": mask}
            results.append(result)
            write_json(
                out / "episodes" / f"final-{policy}-{index:03d}.json",
                {
                    "episode_id": f"final-{policy}-{index:03d}",
                    "done": True,
                    "termination": "submitted",
                    "report": {
                        **result,
                        "training_reward": count / 4,
                        "correct_cells": count,
                        "required_cells": 4,
                    },
                },
            )
        rewards = [r["cell_accuracy"] for r in results]
        finals[policy] = {
            **cell_statistics(results),
            "n": 2,
            "successes": 0,
            "rewards": rewards,
            "mean_reward": sum(rewards) / 2,
            "episode_seed_set": [1, 2],
            "checkpoint": {"artifact_path": "fixture://" + policy},
        }
    summary = render_run_report(
        out,
        {
            "config": {"reward": {"type": "pure_cell"}},
            "status": "complete",
            "attempted_batches": 0,
            "completed_batches": 0,
            "updates": 0,
            "batches": [],
            "final_evaluations": finals,
        },
    )
    assert summary["primary_metric"] == "cell_accuracy"
    assert summary["final"]["base"]["mean_cell_accuracy"] == 0.75
    assert summary["final"]["base"]["std_cell_accuracy"] == 0.25
    assert summary["final"]["base"]["min_cell_accuracy"] == 0.5
    assert summary["final"]["base"]["max_cell_accuracy"] == 1
    assert summary["final_minus_base_mean_cell_accuracy"] == -0.375
    cells = list(csv.DictReader((out / "report/evaluation-cells.csv").open()))
    assert len(cells) == 8 and all(row["rollouts"] == "2" for row in cells)
    comparisons = list(csv.DictReader((out / "report/checkpoint-performance.csv").open()))
    assert (
        len(comparisons) == 2
        and json.loads(comparisons[0]["checkpoint"])["artifact_path"] == "fixture://base"
    )
    assert "Pure cell accuracy" in (out / "report/REPORT.md").read_text()
