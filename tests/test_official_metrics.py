"""Official SRDT counts complete data models, independently of pipeline success."""

from types import SimpleNamespace

import pytest

from elt_rl.errors import InfrastructureError
from elt_rl.metrics import group_metrics
from elt_rl.official_grader import OfficialGrader, parse_target_columns, summarize_official_targets

TARGETS = [SimpleNamespace(name=name, columns=("id", "value")) for name in ("a", "b")]


def table_log(name, passed):
    return (
        f"Table: {name}\n"
        + (
            "Matched columns: ['ID', 'VALUE']\nUnmatched columns: []\n"
            if passed
            else "Matched columns: ['ID']\nUnmatched columns: ['VALUE']\n"
        )
        + "Missed: []\n"
    )


@pytest.mark.parametrize("correct,expected", [(0, 0.0), (1, 0.5), (2, 1.0)])
def test_grader_reports_fractional_srdt_without_changing_whole_pipeline_success(
    tmp_path, monkeypatch, correct, expected
):
    import elt_rl.official_grader as module

    monkeypatch.setattr(module, "task_material", lambda *a: ({}, {"revision": "fixture"}))

    def driver(*args, **kwargs):
        output = tmp_path / "grade/evaluation"
        (output / "driver_errors.json").write_text("[]")
        (output / "agent_results/run/results.log").write_text("Success: fixture schema verified\n")
        (output / "agent_results/run/stage2.log").write_text(
            table_log("a", correct >= 1) + table_log("b", correct == 2)
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", driver)
    reference = tmp_path / "reference.csv"
    reference.write_text("id,value\n1,1\n")
    task = SimpleNamespace(
        name="fixture",
        targets=[SimpleNamespace(**vars(target), expected=reference) for target in TARGETS],
    )
    warehouse = SimpleNamespace(dialect="snowflake", schema="OWNED", database="DB", config={})
    result = OfficialGrader(tmp_path).grade(warehouse, task, tmp_path / "grade")
    assert result["official_srdel"] is True
    assert type(result["official_srdt"]) is float and result["official_srdt"] == expected
    assert result["official_success"] is (correct == 2)
    assert result["correct_target_models"] == correct
    assert result["required_target_models"] == 2


def test_loading_failure_retains_every_target_in_the_denominator():
    parsed = parse_target_columns("", TARGETS, stage1_passed=False)
    metrics = summarize_official_targets(parsed, stage1_passed=False)
    assert metrics["required_target_models"] == 2
    assert metrics["correct_target_models"] == 0
    assert metrics["official_srdt"] == 0.0 and metrics["official_success"] is False


def test_one_target_error_does_not_erase_a_different_complete_target():
    parsed = parse_target_columns(
        table_log("a", True) + "Table: b\nError: object does not exist\n",
        TARGETS,
        stage1_passed=True,
    )
    metrics = summarize_official_targets(parsed, stage1_passed=True)
    assert metrics["official_srdt"] == 0.5 and metrics["official_success"] is False
    with pytest.raises(InfrastructureError, match="Incomplete"):
        parse_target_columns("Table: a\nTable: b\nError: missing\n", TARGETS, stage1_passed=True)


def test_cohort_aggregation_preserves_zero_half_and_one():
    rows = [
        {"reward": 0, "official_srdt": value, "success": value == 1} for value in (0.0, 0.5, 1.0)
    ]
    metrics = group_metrics(rows)
    assert metrics["official_srdt"] == 0.5
    assert metrics["end_to_end_success_rate"] == pytest.approx(1 / 3)


def test_model_counts_micro_average_different_target_counts():
    rows = [
        {
            "reward": 0,
            "official_srdt": 1.0,
            "correct_target_models": 1,
            "required_target_models": 1,
        },
        {
            "reward": 0,
            "official_srdt": 0.0,
            "correct_target_models": 0,
            "required_target_models": 2,
        },
    ]
    assert group_metrics(rows)["official_srdt"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.5, 1.5])
def test_invalid_srdt_cannot_enter_a_cohort(invalid):
    with pytest.raises(ValueError, match="SRDT"):
        group_metrics([{"reward": 0, "official_srdt": invalid}])
