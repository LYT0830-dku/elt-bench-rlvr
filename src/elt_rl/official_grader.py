"""Execute the unmodified original Stage 1/2 drivers against the episode’s warehouse."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.benchmark import task_material
from elt_rl.errors import InfrastructureError


def parse_target_columns(stage2, targets, *, stage1_passed):
    required = {
        target.name.casefold(): {column.upper() for column in target.columns} for target in targets
    }
    results = {
        name: {"matched": [], "unmatched": [], "missed": [], "errors": []} for name in required
    }
    current = next(iter(required)) if len(required) == 1 else None
    for line in stage2.splitlines():
        if line.startswith("Table: "):
            current = line[len("Table: ") :].strip().casefold()
            if current not in required:
                raise InfrastructureError("Official driver compared an undeclared target")
        if line.startswith("Error: "):
            if current is None:
                raise InfrastructureError("Official target error lacks target identity")
            results[current]["errors"].append(line[len("Error: ") :])
        for prefix, key in (
            ("Matched columns: ", "matched"),
            ("Unmatched columns: ", "unmatched"),
            ("Missed: ", "missed"),
        ):
            if line.startswith(prefix):
                if current is None:
                    raise InfrastructureError("Multi-target comparison log lacks target identity")
                values = ast.literal_eval(line[len(prefix) :])
                if not isinstance(values, list) or any(
                    not isinstance(value, str) for value in values
                ):
                    raise InfrastructureError("Malformed official column log")
                results[current][key].extend(values)
    for name, result in results.items():
        observed = result["matched"] + result["unmatched"] + result["missed"]
        if (
            len(observed) != len(set(observed))
            or set(observed) - required[name]
            or (not stage1_passed and observed)
            or (stage1_passed and not result["errors"] and set(observed) != required[name])
        ):
            raise InfrastructureError("Incomplete or inconsistent official comparison log")
    return results


def summarize_official_targets(targets, *, stage1_passed):
    """SRDT counts data models; complete-pipeline success remains a separate boolean.

    Input is the complete, validated per-target output of parse_target_columns.
    Stage 1 failures retain all declared data models in the SRDT denominator.
    """
    if not targets or type(stage1_passed) is not bool:
        raise InfrastructureError("Official metrics require declared targets and loading status")
    passed = {
        name: bool(
            stage1_passed
            and result["matched"]
            and not result["unmatched"]
            and not result["missed"]
            and not result.get("errors")
        )
        for name, result in targets.items()
    }
    correct, total = sum(passed.values()), len(passed)
    return {
        "official_srdel": stage1_passed,
        "official_srdt": correct / total,
        "official_success": stage1_passed and correct == total,
        "correct_target_models": correct,
        "required_target_models": total,
        "target_success": passed,
    }


class OfficialGrader:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()

    def grade(self, warehouse, task, output):
        if warehouse.dialect != "snowflake":
            raise ValueError("This original evaluation adapter requires Snowflake")
        files, lock = task_material(self.repo, task.name)
        root = Path(output).resolve()
        evaluation = root / "evaluation"
        evaluation.mkdir(parents=True)
        for path, content in files.items():
            if path.startswith("evaluation/"):
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
        (root / "elt-bench" / task.name).mkdir(parents=True)
        credential = root / "setup/destination/snowflake_credential.json"
        credential.parent.mkdir(parents=True)
        credential.write_text("{}")
        (evaluation / "agent_results/run").mkdir(parents=True)
        for target in task.targets:
            dest = evaluation / "gt" / task.name / (target.name + ".csv")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target.expected, dest)
        config = {
            "task": task.name,
            "schema": warehouse.schema,
            "database": warehouse.database,
            "connection": warehouse.config,
        }
        private = root / "driver-config.json"
        write_json(private, config)
        os.chmod(private, 0o600)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "elt_rl.grader_worker", str(evaluation), str(private)],
                capture_output=True,
                text=True,
            )
            (root / "driver.stdout.log").write_text(result.stdout)
            (root / "driver.stderr.log").write_text(result.stderr)
            if result.returncode:
                raise InfrastructureError(
                    "Original grader driver failed; inspect host-only driver logs"
                )
        finally:
            private.unlink(missing_ok=True)
        error_file = evaluation / "driver_errors.json"
        if not error_file.is_file():
            raise InfrastructureError("Original grader is missing its typed execution receipt")
        errors = json.loads(error_file.read_text())
        if not isinstance(errors, list) or any(
            not isinstance(error, dict) or type(error.get("infrastructure")) is not bool
            for error in errors
        ):
            raise InfrastructureError("Malformed grader execution receipt")
        if any(error["infrastructure"] for error in errors):
            raise InfrastructureError("Warehouse infrastructure failed during official grading")
        stage1 = (evaluation / "agent_results/run/results.log").read_text()
        stage2 = (
            (evaluation / "agent_results/run/stage2.log").read_text()
            if (evaluation / "agent_results/run/stage2.log").exists()
            else ""
        )
        required = sum(len(t.columns) for t in task.targets)
        stage1_passed = f"Success: {task.name} " in stage1
        targets = parse_target_columns(stage2, task.targets, stage1_passed=stage1_passed)
        matched = [column for result in targets.values() for column in result["matched"]]
        unmatched = [column for result in targets.values() for column in result["unmatched"]]
        missed = [column for result in targets.values() for column in result["missed"]]
        return {
            **summarize_official_targets(targets, stage1_passed=stage1_passed),
            "official_score": len(matched) / required,
            "matched_target_columns": len(matched),
            "required_target_columns": required,
            "score_definition": "derived matched required columns / total required columns; upstream emits logs, not a numeric score",
            "execution": "unmodified original evaluate_stage1 and evaluate_stage2",
            "adapter": "namespace mapping",
            "benchmark_revision": lock["revision"],
            "matched": matched,
            "unmatched": unmatched,
            "missed": missed,
            "targets": targets,
            "stage1_log": stage1,
            "stage2_log": stage2,
        }
