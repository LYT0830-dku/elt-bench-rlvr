"""Official/custom evaluation providers, independent of training reward formulas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elt_rl.grading import compare
from elt_rl.registry import Registry, UnsupportedCapability
from elt_rl.rewards import RewardTable
from elt_rl.warehouse import policy_error


@dataclass(frozen=True)
class EvaluationOutcome:
    metrics: dict
    success: bool
    loading_passed: bool
    predictions: dict[str, RewardTable]
    column_matches: dict[str, dict[str, bool]]
    grades: dict


class OriginalELTBenchmarkEvaluation:
    warehouse_backends = ("snowflake",)

    def evaluate(self, warehouse, task, output, *, grader=None):
        from elt_rl.execution_scoring import load_table
        from elt_rl.official_grader import OfficialGrader

        if warehouse.dialect not in self.warehouse_backends:
            raise UnsupportedCapability(
                "Original evaluation provider requires its supported warehouse backend"
            )
        grader = grader or OfficialGrader(task.runtime["benchmark_repo"])
        metrics = grader.grade(warehouse, task, Path(output) / "official")
        predictions, columns = {}, {}
        for target in task.reward_targets:
            per_target = metrics.get("targets", {}).get(target.name)
            if per_target is None:
                if len(task.reward_targets) != 1:
                    raise ValueError("Multi-target official results must be table-qualified")
                per_target = metrics
            matched = {name.casefold() for name in per_target.get("matched", ())}
            columns[target.name] = {
                field.name: field.name.casefold() in matched for field in target.fields
            }
            path = (
                Path(output)
                / "official/evaluation/agent_results/run"
                / task.name
                / (target.name + ".csv")
            )
            if path.is_file():
                predictions[target.name] = load_table(
                    path, target, max_rows=task.max_reference_rows + 1
                )
        return EvaluationOutcome(
            metrics,
            metrics["official_success"],
            metrics["official_srdel"],
            predictions,
            columns,
            {},
        )


class LocalFixtureEvaluation:
    warehouse_backends = ("duckdb", "snowflake")

    def evaluate(self, warehouse, task, output, *, grader=None):
        grades, predictions, columns = {}, {}, {}
        for target in task.targets:
            try:
                table = warehouse.output(target.name)
                grade = compare(table, target).public()
                predictions[target.name] = RewardTable(table.columns, table.rows)
            except Exception as exc:
                if not policy_error(exc):
                    raise
                grade = {
                    "score": 0,
                    "passed": False,
                    "reason": "Missing or invalid output",
                    "columns": dict.fromkeys(target.columns, False),
                }
            grades[target.name] = grade
            columns[target.name] = grade["columns"]
        success = all(grade["passed"] for grade in grades.values())
        # Local fixtures have no official ELT-Bench SRDEL/SRDT observations.
        return EvaluationOutcome(
            {
                "official_score": None,
                "official_success": None,
                "evaluation_provider": "local_fixture",
                "fixture_success": success,
            },
            success,
            True,
            predictions,
            columns,
            grades,
        )


EVALUATORS = Registry("evaluation provider")
EVALUATORS.register("original_elt_bench", OriginalELTBenchmarkEvaluation())
EVALUATORS.register("local_fixture", LocalFixtureEvaluation())
