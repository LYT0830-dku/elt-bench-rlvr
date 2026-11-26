"""Correct historical SRDT from original logs; no model, warehouse or source-file writes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.errors import IntegrityError
from elt_rl.official_grader import parse_target_columns, summarize_official_targets


def reaggregate(run, task, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError("Write corrected metrics outside the immutable original run")
    state = json.loads((run / "protocol-state.json").read_text())
    cohorts = [
        *(batch["rollout"] for batch in state["batches"]),
        *state.get("process_evaluations", []),
        *state.get("final_evaluations", {}).values(),
    ]
    rows, groups = [], []
    for group in cohorts:
        members = []
        if len(group["episodes"]) != group["n"]:
            raise IntegrityError("Historical cohort is incomplete")
        for row in group["episodes"]:
            identifier, evidence = row["episode_id"], row["artifact"]
            path = run / "episodes" / (identifier + ".json")
            path = path if path.is_file() else Path(evidence["artifact_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != evidence["sha256"]:
                raise IntegrityError("Historical trajectory digest differs: " + identifier)
            raw = json.loads(path.read_text())
            if raw.get("episode_id") != identifier or raw.get("task") != task.id:
                raise IntegrityError("Historical episode/task identity differs")
            report = raw.get("report") or {}
            if not raw.get("done") or str(raw.get("termination", "")).startswith("infrastructure"):
                raise IntegrityError("An incomplete/infrastructure attempt is not a policy outcome")
            if report:
                if not report.get("training_eligible", True):
                    raise IntegrityError("Ineligible historical scoring cannot enter a cohort")
                stage1 = report.get("official_srdel")
                if stage1 is None:
                    stage1 = f"Success: {task.id} " in report.get("stage1_log", "")
                if type(stage1) is not bool:
                    raise IntegrityError("Historical loading status is malformed")
                stage2 = report.get("stage2_log")
                if stage2 is None:
                    raise IntegrityError("Original Stage 2 log is unavailable: " + identifier)
                compared = parse_target_columns(stage2, task.targets, stage1_passed=stage1)
                corrected = summarize_official_targets(compared, stage1_passed=stage1)
            else:
                if row["reward"] != 0 or raw.get("termination") in {"cancelled", "user_pause"}:
                    raise IntegrityError("Only a known policy-zero may lack a scoring report")
                corrected = summarize_official_targets(
                    {t.name: {"matched": [], "unmatched": [], "missed": []} for t in task.targets},
                    stage1_passed=False,
                )
            selected = {
                "episode_id": identifier,
                "cohort": group["label"],
                "reward": row["reward"],
                "previous_srdt": report.get("official_srdt", report.get("official_success", False)),
                "official_srdt": corrected["official_srdt"],
                "official_srdel": corrected["official_srdel"],
                "end_to_end_success": corrected["official_success"],
                "correct_target_models": corrected["correct_target_models"],
                "required_target_models": corrected["required_target_models"],
                "target_success": corrected["target_success"],
                "known_policy_zero": not bool(report),
                "original_episode_sha256": digest,
            }
            members.append(selected)
            rows.append(selected)
        correct = sum(r["correct_target_models"] for r in members)
        total = sum(r["required_target_models"] for r in members)
        groups.append(
            {
                "cohort": group["label"],
                "n": group["n"],
                "previous_srdt": group.get("official_srdt"),
                "official_srdt": correct / total,
                "official_srdel": sum(r["official_srdel"] for r in members) / len(members),
                "end_to_end_success_rate": sum(r["end_to_end_success"] for r in members)
                / len(members),
                "correct_target_models": correct,
                "required_target_models": total,
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "cohorts.json", groups)
    write_json(output / "rollouts.json", rows)
    changed = [r for r in rows if r["previous_srdt"] != r["official_srdt"]]
    result = {
        "status": "complete",
        "task": task.id,
        "selected_rollouts": len(rows),
        "changed_rollouts": len(changed),
        "changes": changed,
        "reward_changed": False,
        "original_records_modified": False,
        "method": "Offline aggregation of unmodified pinned official per-target column logs; all declared target models remain in the denominator",
        "source_protocol_sha256": hashlib.sha256(
            (run / "protocol-state.json").read_bytes()
        ).hexdigest(),
    }
    write_json(output / "correction-summary.json", result)
    with (output / "rollouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()} for row in rows
        )
    return result


if __name__ == "__main__":
    from elt_rl.task_spec import load_task_spec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = reaggregate(args.run, load_task_spec(args.task_spec), args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "changes"}))
