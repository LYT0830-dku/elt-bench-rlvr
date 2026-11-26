"""Offline pure-cell rescoring of real saved tables; never sample or replay ELT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from elt_rl.artifacts import valid_submission_receipt, write_json
from elt_rl.execution_scoring import load_table
from elt_rl.metrics import cell_statistics
from elt_rl.rewards import RewardConfig, RewardContext, score_execution
from elt_rl.task_spec import load_task_spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rescore_run(run, task, output, *, evidence_manifest=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    task.verify()
    manifest = json.loads(Path(evidence_manifest).read_text()) if evidence_manifest else None
    manifest_root = Path(evidence_manifest).resolve().parent if evidence_manifest else None
    state = json.loads((run / "protocol-state.json").read_text())
    completed = {}
    cohorts = [
        *(b["rollout"] for b in state.get("batches", [])),
        *state.get("process_evaluations", []),
        *state.get("final_evaluations", {}).values(),
    ]
    for cohort in cohorts:
        for row in cohort["episodes"]:
            completed[row["episode_id"]] = row
    records, cells, observations = [], [], {}

    def checked(path, expected=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("Saved actual/reference snapshot is unavailable: " + str(path))
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError("Saved evidence digest differs: " + str(path))
        return {"artifact_path": str(path), "sha256": actual}

    for path in sorted((run / "episodes").glob("*.json")):
        raw, raw_ref = json.loads(path.read_text()), checked(path)
        identifier = raw.get("episode_id", path.stem)
        if identifier in completed:
            checked(path, completed[identifier].get("artifact", {}).get("sha256"))
        if raw.get("task") is not None and raw["task"] not in {task.id, task.execution_task.name}:
            raise ValueError("Saved rollout belongs to a different task")
        report = raw.get("report") or {}
        record = {
            "episode_id": identifier,
            "cohort": identifier.rsplit("-", 1)[0],
            "selected_in_original_cohort": identifier in completed,
            "recorded_reward": report.get("training_reward"),
            "source": raw_ref,
        }
        try:
            delivered = raw.get("termination") == "submitted"
            cancelled_after_grading = raw.get(
                "termination"
            ) == "infrastructure_CancelledError" and valid_submission_receipt(raw)
            if raw.get("done") is not True or not (delivered or cancelled_after_grading):
                raise FileNotFoundError(
                    "No completed submitted warehouse prediction; not reconstructed"
                )
            record["delivery_cancelled_after_grading"] = cancelled_after_grading
            if (
                report.get("pipeline_valid") is not True
                or report.get("execution_success") is not True
            ):
                raise FileNotFoundError("No verified loading/submission execution receipt")
            snapshots = None
            receipt = (report.get("reward_result") or {}).get("artifact")
            if receipt:
                checked(receipt["artifact_path"], receipt["sha256"])
                snapshots = json.loads(Path(receipt["artifact_path"]).read_text())[
                    "targets_snapshot"
                ]
            predictions, references, sources = {}, {}, {}
            for target in task.reward_targets:
                if snapshots is not None:
                    snapshot = snapshots[target.name]
                    refs = [snapshot.get(kind) for kind in ("prediction", "reference")]
                    if any(ref is None for ref in refs):
                        raise FileNotFoundError(
                            "Actual prediction/reference snapshot was not saved"
                        )
                    for kind, ref in zip(("prediction", "reference"), refs, strict=True):
                        sources[target.name + "." + kind] = checked(
                            ref["artifact_path"], ref["sha256"]
                        )
                else:
                    if manifest is None:
                        raise FileNotFoundError(
                            "Historical snapshots require their original evidence manifest"
                        )
                    evaluation = (
                        path.with_name(path.stem + "-files")
                        / "submission-result/official/evaluation"
                    )
                    for kind, prefix in [("prediction", "agent_results/run"), ("reference", "gt")]:
                        saved = (
                            evaluation / prefix / task.execution_task.name / (target.name + ".csv")
                        )
                        expected = manifest.get(str(saved.relative_to(manifest_root)))
                        if expected is None:
                            raise FileNotFoundError(
                                "Historical snapshot has no original digest: " + str(saved)
                            )
                        sources[target.name + "." + kind] = checked(saved, expected)
                for kind, destination in [("prediction", predictions), ("reference", references)]:
                    destination[target.name] = load_table(
                        sources[target.name + "." + kind]["artifact_path"],
                        target,
                        max_rows=task.max_reference_rows,
                    )
            result = score_execution(
                RewardContext(task.id, task.reward_targets, predictions, references, {}, True, True)
            )
            observations[identifier] = result
            record.update(
                status="rescored_from_actual_tables",
                cell_accuracy=result["cell_accuracy"],
                correct_cells=result["correct_cells"],
                required_cells=result["required_cells"],
                snapshots=sources,
            )
            for target, details in result["targets"].items():
                cells.extend(
                    {"episode_id": identifier, "target": target, **cell}
                    for cell in details["cells"]
                )
        except (FileNotFoundError, KeyError) as exc:
            record.update(status="unavailable", reason=str(exc), cell_accuracy=None)
        records.append(record)
    batch_rows = []
    from tinker_cookbook.rl.data_processing import compute_advantages

    for batch in state.get("batches", []):
        ids = [r["episode_id"] for r in batch["rollout"]["episodes"]]
        values = [observations[i]["reward"] for i in ids if i in observations]
        item = {
            "batch": batch["batch"],
            "original_rewards": batch["rewards"],
            "original_advantages": batch["advantages"],
            "original_status": batch["status"],
            "original_reward_variance": statistics.pvariance(batch["rewards"]),
            "offline_only": True,
            "optimizer_updated_by_rescoring": False,
        }
        if len(values) == len(ids):
            advantages = compute_advantages(
                [SimpleNamespace(get_total_rewards=lambda values=values: values)]
            )[0]
            if max(values) == min(values):
                advantages.zero_()
            item.update(
                status="rescored",
                cell_rewards=values,
                offline_advantages=advantages.tolist(),
                cell_reward_variance=statistics.pvariance(values),
                nonzero_learning_signal=bool((advantages > 0).any() and (advantages < 0).any()),
            )
        else:
            item.update(
                status="unavailable", missing_episodes=[i for i in ids if i not in observations]
            )
        batch_rows.append(item)
    summaries = []
    for label in sorted({r["cohort"] for r in records}):
        selected = [r for r in records if r["cohort"] == label]
        usable = [
            observations[r["episode_id"]] for r in selected if r["episode_id"] in observations
        ]
        summaries.append(
            {
                "cohort": label,
                "artifacts": len(selected),
                "rescored": len(usable),
                "unavailable": len(selected) - len(usable),
                "complete_original_cohort": all(r["selected_in_original_cohort"] for r in selected),
                **cell_statistics(usable),
            }
        )
    result = {
        "reward_configuration": RewardConfig().configuration(),
        "offline_only": True,
        "task": task.id,
        "run": str(run),
        "episodes": records,
        "batches": batch_rows,
        "cohorts": summaries,
        "rescored": len(observations),
        "unavailable": len(records) - len(observations),
        "nonzero_learning_signal_groups": sum(
            b.get("nonzero_learning_signal", False) for b in batch_rows
        ),
    }
    write_json(output / "rescores.json", result)
    write_json(output / "batch-comparison.json", batch_rows)
    for name, rows in [("episodes", records), ("cells", cells)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                for row in rows
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path)
    args = parser.parse_args()
    result = rescore_run(
        args.run,
        load_task_spec(args.task_spec),
        args.output,
        evidence_manifest=args.evidence_manifest,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["task", "rescored", "unavailable", "nonzero_learning_signal_groups"]
            }
        )
    )
