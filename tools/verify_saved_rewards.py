"""Recompute the execution reward from saved trusted tables without replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from elt_rl.artifacts import implementation_fingerprint, write_json
from elt_rl.rewards import (
    NumericReward,
    RewardConfig,
    RewardContext,
    RewardField,
    RewardTable,
    RewardTarget,
    score_execution,
)


def verify(path, *, reference_task_spec=None):
    path = Path(path)
    data = json.loads(path.read_text())
    targets = []
    for target in data["targets"]:
        fields = []
        for raw in target["fields"]:
            raw = dict(raw)
            if raw.get("numeric") is not None:
                raw["numeric"] = NumericReward(**raw["numeric"])
            fields.append(RewardField(**raw))
        targets.append(RewardTarget(target["name"], tuple(fields), tuple(target["keys"])))
    configuration = RewardConfig.from_dict(data["configuration"])
    external_references = None
    if reference_task_spec is not None:
        from elt_rl.execution_scoring import load_table
        from elt_rl.task_spec import load_task_spec

        task = load_task_spec(reference_task_spec)
        if task.id != data["task_id"] or tuple(task.reward_targets) != tuple(targets):
            raise ValueError("External reference task differs from the recorded reward contract")
        declared = {target.name: target for target in task.targets}
        external_references = {
            target.name: load_table(
                declared[target.name].expected, target, max_rows=task.max_reference_rows
            )
            for target in task.reward_targets
        }
    results = []
    for episode in data["episodes"]:

        def tables(key, source=episode):
            return {
                name: RewardTable(
                    tuple(table["columns"]), tuple(tuple(row) for row in table["rows"])
                )
                for name, table in source[key].items()
            }

        context = RewardContext(
            data["task_id"],
            tuple(targets),
            tables("predictions"),
            external_references if external_references is not None else tables("references"),
            episode["official_columns"],
            episode["stage1_passed"],
            episode["submission_valid"],
        )
        result = score_execution(context, configuration)
        results.append(
            {
                "episode": episode["episode"],
                "recorded_reward": episode["reward"],
                "recomputed_reward": result["reward"],
                "absolute_difference": abs(result["reward"] - episode["reward"]),
                **{
                    key: result[key]
                    for key in (
                        "cell_accuracy",
                        "correct_cells",
                        "required_cells",
                        "column_score",
                        "cell_score",
                        "value_score",
                    )
                    if key in result
                },
            }
        )
    if not results:
        raise ValueError("Saved context file contains no completed execution results")
    return {
        "status": "passed"
        if all(row["absolute_difference"] == 0 for row in results)
        else "mismatch",
        "episodes": results,
        "completed_actual_episodes": len(results),
        "maximum_absolute_difference": max(row["absolute_difference"] for row in results),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": implementation_fingerprint(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-task-spec",
        type=Path,
        help="Read verified references from this private task instead of redistributing them",
    )
    arguments = parser.parse_args()
    receipt = verify(arguments.input, reference_task_spec=arguments.reference_task_spec)
    write_json(arguments.output, receipt)
    print(
        json.dumps(
            {
                key: value
                for key, value in receipt.items()
                if key not in {"episodes", "implementation_sha256"}
            }
        )
    )
    raise SystemExit(0 if receipt["status"] == "passed" else 1)
