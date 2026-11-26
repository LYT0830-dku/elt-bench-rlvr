"""Private snapshots and reward inputs built from actual warehouse/driver results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.lineage import evidence_ref, read_evidence
from elt_rl.reward_registry import REWARDS
from elt_rl.rewards import (
    REWARD_METRICS,
    RewardConfig,
    RewardContext,
    RewardInfrastructureError,
    RewardTable,
)
from elt_rl.task_spec import _csv_value


def load_table(path, target, *, max_rows):
    fields = {field.name.casefold(): field for field in target.fields}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        columns = tuple(next(reader))
        if len(columns) != len({column.casefold() for column in columns}):
            raise RewardInfrastructureError("Duplicate columns in trusted table snapshot")
        rows = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                raise RewardInfrastructureError("Table snapshot exceeds configured host row budget")
            rows.append(
                tuple(
                    _csv_value(value, fields[column.casefold()])
                    if column.casefold() in fields
                    else value
                    for column, value in zip(columns, row, strict=True)
                )
            )
    return RewardTable(columns, tuple(rows))


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(table.columns)
        writer.writerows(table.rows)
    path.chmod(0o600)


def score_submission(task, output, outcome, configuration, *, loading_succeeded=None):
    task.verify()
    configuration = configuration or RewardConfig().configuration()
    pure_cell = configuration.get("type", "pure_cell") == "pure_cell"
    if pure_cell and type(loading_succeeded) is not bool:
        raise RewardInfrastructureError(
            "Pure cell scoring requires a verified loading execution state"
        )
    trusted = Path(output) / "trusted-scoring"
    trusted.mkdir(parents=True, exist_ok=True, mode=0o700)
    trusted.chmod(0o700)
    targets = {target.name: target for target in task.reward_targets}
    references, snapshots = {}, {}
    for target_contract in task.targets:
        target = targets[target_contract.name]
        reference_path = trusted / (target_contract.name + ".reference.csv")
        reference_path.write_bytes(target_contract.expected.read_bytes())
        reference_path.chmod(0o600)
        references[target_contract.name] = load_table(
            reference_path, target, max_rows=task.max_reference_rows
        )
        prediction_path = trusted / (target_contract.name + ".prediction.csv")
        if target_contract.name in outcome.predictions:
            save_table(prediction_path, outcome.predictions[target_contract.name])
        snapshots[target_contract.name] = {
            "reference": evidence_ref(reference_path),
            "prediction": evidence_ref(prediction_path) if prediction_path.is_file() else None,
            "keys": target.keys,
            "columns": [field.name for field in target.fields],
        }
    context = RewardContext(
        task.id,
        task.reward_targets,
        outcome.predictions,
        references,
        outcome.column_matches,
        loading_succeeded if pure_cell else outcome.loading_passed,
        True,
    )
    result = REWARDS.score(context, configuration)
    selection_sha = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    result.update(task_id=task.id, targets_snapshot=snapshots, selection_sha256=selection_sha)
    destination = trusted / "reward-result.json"
    write_json(destination, result)
    return result, {
        "artifact": evidence_ref(destination),
        "reward": result["reward"],
        "reward_type": result["reward_type"],
        "config_sha256": result["config_sha256"],
        "selection_sha256": selection_sha,
        **{key: result[key] for key in REWARD_METRICS if key in result},
        **({"cell_results": result["cell_results"]} if "cell_results" in result else {}),
        "snapshots_sha256": hashlib.sha256(str(snapshots).encode()).hexdigest(),
    }


def verify_scoring_receipt(raw, configuration):
    configuration = configuration or RewardConfig().configuration()
    report = raw.get("report") or {}
    receipt = report.get("reward_result")
    try:
        expected_type = RewardConfig.from_dict(configuration).type
    except (ValueError, TypeError) as exc:
        raise RewardInfrastructureError(
            "Trajectory does not use the frozen selected reward"
        ) from exc
    if not isinstance(receipt, dict) or report.get("reward_type") != expected_type:
        raise RewardInfrastructureError("Trajectory does not use the frozen selected reward")
    result = read_evidence(receipt["artifact"])
    expected_selection = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    if (
        result.get("selection_sha256") != expected_selection
        or receipt.get("selection_sha256") != expected_selection
    ):
        raise RewardInfrastructureError(
            "Trusted score uses a different frozen reward configuration"
        )
    if (
        result.get("config_sha256") != receipt.get("config_sha256")
        or result.get("reward") != report.get("training_reward")
        or result.get("reward_type") != report.get("reward_type")
    ):
        raise RewardInfrastructureError("Trusted reward result/trajectory receipt mismatch")
    if result.get("reward_type") == "pure_cell" and (
        report.get("cell_results") != result.get("cell_results")
        or report.get("cell_accuracy") != result["reward"]
        or result["correct_cells"] != sum(cell["correct"] for cell in result["cell_results"])
        or result["required_cells"] != len(result["cell_results"])
    ):
        raise RewardInfrastructureError("Pure cell counts or trajectory cell results differ")
    for snapshot in result.get("targets_snapshot", {}).values():
        for kind in ("prediction", "reference"):
            saved = snapshot.get(kind)
            if saved is not None and evidence_ref(Path(saved["artifact_path"])) != saved:
                raise RewardInfrastructureError("Saved actual/reference table fingerprint changed")
    return result
