"""Pure cell accuracy, trusted alignment and evaluation statistics contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from elt_rl.execution_scoring import score_submission, verify_scoring_receipt
from elt_rl.metrics import cell_statistics
from elt_rl.rewards import (
    RewardConfig,
    RewardContext,
    RewardField,
    RewardInfrastructureError,
    RewardTable,
    RewardTarget,
    score_execution,
)


def example(rows=None):
    target = RewardTarget(
        "output",
        (
            RewardField("id", "integer", "identifier"),
            RewardField("label"),
            RewardField("amount", "decimal", comparator="relative"),
        ),
        ("id",),
    )
    table = RewardTable(("id", "label", "amount"), ((1, "one", "100"), (2, "two", "200")))
    return RewardContext(
        "generic",
        (target,),
        {"output": table if rows is None else replace(table, rows=tuple(rows))},
        {"output": table},
        {},
        True,
        True,
    )


def test_default_is_binary_cell_accuracy_with_no_official_column_dependency():
    value = example(((1, "wrong", "90"), (2, "two", "200")))
    result = score_execution(value)
    assert result["reward_type"] == "pure_cell"
    assert result["reward"] == result["cell_accuracy"] == 4 / 6
    assert result["correct_cells"] == 4 and result["required_cells"] == 6
    assert "column_score" not in result and "value_score" not in result
    assert set(c["value"] for c in result["targets"]["output"]["cells"]) == {0, 1}
    changed = replace(
        value, official_columns={"output": {"id": False, "label": True, "amount": True}}
    )
    assert score_execution(changed) == result


def test_missing_row_receives_partial_credit_with_fixed_denominator():
    result = score_execution(example(((1, "one", "100"),)))
    assert result["reward"] == 0.5
    assert result["correct_cells"] == 3 and result["required_cells"] == 6
    missing = [c for c in result["targets"]["output"]["cells"] if c["key"] == ["2"]]
    assert len(missing) == 3 and all(c["missing"] and not c["correct"] for c in missing)


def test_missing_column_and_empty_prediction_keep_reference_denominator():
    value = example()
    value = replace(
        value, predictions={"output": RewardTable(("id", "label"), ((1, "one"), (2, "two")))}
    )
    assert score_execution(value)["reward"] == 4 / 6
    empty = replace(value, predictions={"output": RewardTable(("id", "label"), ())})
    assert score_execution(empty)["reward"] == 0
    assert score_execution(empty)["required_cells"] == 6


@pytest.mark.parametrize(
    "rows",
    [
        ((1, "one", "100"), (1, "one", "100")),
        ((1, "one", "100"), (2, "two", "200"), (3, "extra", "0")),
        ((None, "one", "100"), (2, "two", "200")),
        ((True, "one", "100"), (2, "two", "200")),
    ],
)
def test_duplicate_extra_null_or_invalid_keys_cannot_gain_credit(rows):
    result = score_execution(example(rows))
    assert result["reward"] == 0 and result["required_cells"] == 6
    assert result["targets"]["output"]["error"]


def multi_target():
    value = example()
    small = RewardTarget(
        "small", (RewardField("id", "integer", "identifier"), RewardField("label")), ("id",)
    )
    return replace(
        value,
        targets=(*value.targets, small),
        predictions={**value.predictions, "small": RewardTable(("id", "label"), ((10, "wrong"),))},
        references={**value.references, "small": RewardTable(("id", "label"), ((10, "right"),))},
    )


def test_multi_table_uses_micro_average_and_distinct_cell_identities():
    result = score_execution(multi_target())
    assert result["reward"] == 7 / 8  # Macro average would incorrectly give .75.
    assert result["correct_cells"] == 7 and result["required_cells"] == 8
    assert {c["target"] for c in result["cell_results"]} == {"output", "small"}


def test_missing_or_rejected_target_does_not_erase_other_target_credit():
    value = multi_target()
    result = score_execution(replace(value, predictions={"small": value.predictions["small"]}))
    assert result["reward"] == 1 / 8
    assert result["required_cells"] == 8


def test_numeric_tolerance_is_binary_and_distance_has_no_reward_effect(monkeypatch):
    import elt_rl.rewards as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure cell accuracy must never calculate numeric proximity")

    monkeypatch.setattr(module, "_value", forbidden)
    assert score_execution(example(((1, "one", "99"), (2, "two", "202"))))["reward"] == 1
    near = score_execution(example(((1, "one", "98.9"), (2, "two", "200"))))
    far = score_execution(example(((1, "one", "0"), (2, "two", "200"))))
    assert near["reward"] == far["reward"] == 5 / 6


def test_composite_keys_reordering_and_missing_rows():
    target = RewardTarget(
        "composite",
        (
            RewardField("country", role="identifier"),
            RewardField("id", "integer", "identifier"),
            RewardField("label"),
        ),
        ("country", "id"),
    )
    table = RewardTable(("country", "id", "label"), (("a", 1, "one"), ("b", 1, "two")))
    value = RewardContext(
        "generic",
        (target,),
        {target.name: replace(table, rows=table.rows[::-1])},
        {target.name: table},
        {},
        True,
        True,
    )
    assert score_execution(value)["reward"] == 1
    assert (
        score_execution(
            replace(value, predictions={target.name: replace(table, rows=table.rows[:1])})
        )["reward"]
        == 0.5
    )


@pytest.mark.parametrize("gate", ["stage1_passed", "submission_valid"])
def test_failed_execution_gate_has_complete_zero_mask_but_unknown_is_infrastructure(gate):
    result = score_execution(replace(example(), **{gate: False}))
    assert result["reward"] == 0 and len(result["cell_results"]) == 6
    with pytest.raises(RewardInfrastructureError):
        score_execution(replace(example(), **{gate: None}))


def test_broken_reference_is_not_a_policy_zero():
    with pytest.raises(RewardInfrastructureError):
        score_execution(replace(example(), references={}))


@pytest.mark.parametrize(
    "configuration",
    [
        {"weights": [0, 1, 0]},
        {"numeric": {"alpha": 5}},
        {"type": "pure_cell", "version": "execution_reward_v1"},
    ],
)
def test_pure_cell_rejects_mixture_proximity_and_version_confusion(configuration):
    with pytest.raises(ValueError):
        RewardConfig.from_dict(configuration)


@pytest.mark.parametrize(
    "name", ["column_execution", "column_cell_execution", "hierarchical_execution", "execution"]
)
def test_historical_baselines_require_explicit_selection_and_keep_their_semantics(name):
    value = example(((1, "one", "100"),))
    config = RewardConfig.from_dict({"type": name})
    assert score_execution(value, config)["reward"] == 0  # Historical missing-row rule.
    assert score_execution(value)["reward"] == 0.5


def test_evaluation_reports_population_statistics_and_every_cell():
    outputs = [score_execution(example()), score_execution(example(((1, "one", "100"),)))]
    result = cell_statistics(outputs)
    assert result["mean_cell_accuracy"] == 0.75
    assert result["std_cell_accuracy"] == 0.25
    assert result["min_cell_accuracy"] == 0.5 and result["max_cell_accuracy"] == 1
    assert len(result["per_cell_accuracy"]) == 6
    cell = next(c for c in result["per_cell_accuracy"] if c["key"] == ["2"] and c["column"] == "id")
    assert cell["accuracy"] == 0.5 and cell["rollouts"] == 2


def test_pure_submission_uses_actual_loading_gate_not_official_srdel(tmp_path):
    value = example()
    expected = tmp_path / "reference.csv"
    expected.write_text("id,label,amount\n1,one,100\n2,two,200\n")
    task = SimpleNamespace(
        id="generic",
        verify=lambda: None,
        reward_targets=value.targets,
        targets=(SimpleNamespace(name="output", expected=expected),),
        max_reference_rows=100,
    )
    outcome = SimpleNamespace(
        predictions=value.predictions, column_matches={}, loading_passed=False
    )
    config = RewardConfig().configuration()
    result, receipt = score_submission(
        task, tmp_path / "scoring", outcome, config, loading_succeeded=True
    )
    assert result["reward"] == 1
    raw = {
        "report": {
            "reward_type": "pure_cell",
            "training_reward": 1,
            "cell_accuracy": 1,
            "cell_results": result["cell_results"],
            "reward_result": receipt,
        }
    }
    assert verify_scoring_receipt(raw, config)["reward"] == 1
    raw["report"]["cell_results"] = []
    with pytest.raises(RewardInfrastructureError):
        verify_scoring_receipt(raw, config)
    with pytest.raises(RewardInfrastructureError):
        score_submission(task, tmp_path / "unknown", outcome, config)
