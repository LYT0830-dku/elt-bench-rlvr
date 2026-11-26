from dataclasses import replace
from decimal import Decimal

import pytest

from elt_rl.rewards import (
    NumericReward,
    RewardConfig,
    RewardContext,
    RewardField,
    RewardInfrastructureError,
    RewardTable,
    RewardTarget,
    score_execution,
)


def context(rows=None, *, target=None, reference=None, columns=None, official=None):
    target = target or RewardTarget(
        "output",
        (
            RewardField("id", "integer", "identifier"),
            RewardField("label", "string"),
            RewardField("amount", "decimal", comparator="relative"),
        ),
        ("id",),
    )
    columns = columns or tuple(f.name for f in target.fields)
    expected = reference or ((1, "one", "100.00"), (2, "two", "200.00"))
    return RewardContext(
        "fixture",
        (target,),
        {target.name: RewardTable(columns, tuple(rows if rows is not None else expected))},
        {target.name: RewardTable(tuple(f.name for f in target.fields), tuple(expected))},
        {target.name: official or {f.name: True for f in target.fields}},
        True,
        True,
    )


def score(value):
    return score_execution(value, RewardConfig(type="execution"))


def test_correct_full_score_repeatable_and_row_order_invariant():
    value = context()
    result = score(value)
    assert (
        result["reward"] == result["column_score"] == result["cell_score"] == result["reward"] == 1
    )
    assert result == score(value)
    prediction = value.predictions["output"]
    reordered = replace(
        value, predictions={"output": replace(prediction, rows=prediction.rows[::-1])}
    )
    assert score(reordered) == result


@pytest.mark.parametrize(
    "rows",
    [
        ((1, "one", "100"),),
        ((1, "one", "100"), (1, "one", "100")),
        ((1, "one", "100"), (2, "two", "200"), (3, "x", "1")),
        ((None, "one", "100"), (2, "two", "200")),
    ],
)
def test_invalid_key_sets_cannot_increase_any_reward(rows):
    result = score(context(rows))
    assert result["column_score"] == result["cell_score"] == result["reward"] == 0
    assert result["targets"]["output"]["error"]


def test_missing_column_keeps_full_required_denominator():
    value = context(
        ((1, "one"), (2, "two")),
        columns=("id", "label"),
        official={"id": True, "label": True, "amount": False},
    )
    result = score(value)
    assert (
        result["column_score"]
        == result["cell_score"]
        == result["value_score"]
        == pytest.approx(2 / 3)
    )
    assert result["targets"]["output"]["required_cells"] == 6
    assert sum(cell["missing"] for cell in result["targets"]["output"]["cells"]) == 2


def test_same_column_score_different_cells_and_continuous_values():
    first = context(
        ((1, "wrong", "90"), (2, "two", "200")),
        official={"id": True, "label": False, "amount": False},
    )
    second = context(
        ((1, "wrong", "0"), (2, "wrong", "0")),
        official={"id": True, "label": False, "amount": False},
    )
    a, b = score(first), score(second)
    assert a["column_score"] == b["column_score"] == 1 / 3
    assert a["cell_score"] > b["cell_score"] and a["reward"] > b["reward"]
    closer = replace(
        first,
        predictions={
            "output": RewardTable(
                ("id", "label", "amount"), ((1, "wrong", "95"), (2, "two", "200"))
            )
        },
    )
    assert score(closer)["value_score"] > a["value_score"]


def test_multi_target_equal_weight_and_same_column_names_are_separate():
    first = context()
    second = RewardTarget(
        "other", (RewardField("id", "integer", "identifier"), RewardField("label")), ("id",)
    )
    value = replace(
        first,
        targets=(*first.targets, second),
        predictions={**first.predictions, "other": RewardTable(("id", "label"), ((10, "wrong"),))},
        references={**first.references, "other": RewardTable(("id", "label"), ((10, "right"),))},
        official_columns={**first.official_columns, "other": {"id": True, "label": False}},
    )
    result = score(value)
    assert result["column_score"] == result["cell_score"] == result["value_score"] == 0.75
    assert result["targets"]["output"]["columns"]["label"]["official_match"]
    assert not result["targets"]["other"]["columns"]["label"]["official_match"]


@pytest.mark.parametrize("gate", ["stage1_passed", "submission_valid"])
def test_policy_execution_gate_zero_but_unknown_gate_is_infrastructure(gate):
    assert score(replace(context(), **{gate: False}))["reward"] == 0
    with pytest.raises(RewardInfrastructureError):
        score(replace(context(), **{gate: None}))


def test_broken_reference_and_infrastructure_fault_do_not_become_zero():
    value = context()
    for references in (
        {},
        {"output": RewardTable(("id", "label", "amount"), ((1, "x", "NaN"),))},
        {"output": RewardTable(("id", "label", "amount"), ((1, "x", "1"), (1, "y", "2")))},
    ):
        with pytest.raises(RewardInfrastructureError):
            score(replace(value, references=references))
    with pytest.raises(RewardInfrastructureError):
        score(replace(value, infrastructure_error="Warehouse unavailable"))


@pytest.mark.parametrize("illegal", ["NaN", "Inf", "-Inf", "invalid", True, None])
def test_invalid_numeric_prediction_cannot_gain_proximity(illegal):
    value = context(
        ((1, "one", illegal), (2, "two", "200")),
        official={"id": True, "label": True, "amount": False},
    )
    cell = score(value)["targets"]["output"]["cells"][4]
    assert cell["column"] == "amount" and cell["value"] == 0 and not cell["correct"]


def test_zero_negative_large_numeric_and_exact_identifier_types():
    fields = (
        RewardField("id", "integer", "identifier"),
        RewardField("amount", "decimal", comparator="relative"),
    )
    target = RewardTarget("numbers", fields, ("id",))
    large = 9007199254740993
    value = context(
        ((large, "0"), (large + 1, "-100")),
        target=target,
        reference=((large, "0"), (large + 1, "-100")),
    )
    assert score(value)["reward"] == 1
    mismatched = replace(
        value,
        predictions={
            "numbers": RewardTable(("id", "amount"), ((large - 1, "0"), (large + 1, "-100")))
        },
    )
    assert score(mismatched)["reward"] == 0
    bad_zero = replace(
        value,
        predictions={
            "numbers": RewardTable(("id", "amount"), ((large, "0.1"), (large + 1, "-90")))
        },
        official_columns={"numbers": {"id": True, "amount": False}},
    )
    cells = score(bad_zero)["targets"]["numbers"]["cells"]
    assert cells[2]["value"] == pytest.approx(__import__("math").exp(-0.5))
    assert cells[3]["value"] == pytest.approx(__import__("math").exp(-0.45))


def test_identifier_and_classification_codes_do_not_get_numeric_distance():
    target = RewardTarget(
        "coded",
        (
            RewardField("id", "integer", "identifier"),
            RewardField("code", "integer", "categorical", comparator="relative"),
        ),
        ("id",),
    )
    value = context(
        ((1, 1001),), target=target, reference=((1, 1000),), official={"id": True, "code": False}
    )
    assert score(value)["value_score"] == 0.5


def test_dates_booleans_null_and_legal_count_domain():
    target = RewardTarget(
        "typed",
        (
            RewardField("id", "string", "identifier"),
            RewardField("day", "date"),
            RewardField("enabled", "boolean"),
            RewardField("optional", "decimal", nullable=True),
            RewardField("count", "integer", minimum="0"),
        ),
        ("id",),
    )
    expected = (("001", "2026-11-26", True, None, "2"),)
    value = context(expected, target=target, reference=expected)
    assert score(value)["reward"] == 1
    for count in ("-1", "1.5"):
        bad = replace(
            value,
            predictions={
                "typed": RewardTable(
                    tuple(f.name for f in target.fields),
                    (("001", "2026-11-26", "true", None, count),),
                )
            },
            official_columns={"typed": {f.name: f.name != "count" for f in target.fields}},
        )
        assert score(bad)["targets"]["typed"]["cells"][-1]["value"] == 0


def test_official_negative_boundary_is_reported_without_changing_column_result():
    target = RewardTarget(
        "negative",
        (
            RewardField("id", "integer", "identifier"),
            RewardField("amount", "decimal", comparator="official_elt_relative_v1"),
        ),
        ("id",),
    )
    value = context(
        ((1, "-10"),),
        target=target,
        reference=((1, "-10"),),
        official={"id": True, "amount": False},
    )
    result = score(value)
    assert result["column_score"] == 0.5
    cell = result["targets"]["negative"]["cells"][-1]
    assert not cell["correct"] and cell["value"] == 1 and cell["custom_value_comparator_difference"]


def test_officially_accepted_legal_numeric_cell_always_gets_value_one():
    value = context(((1, "one", "101"), (2, "two", "198")))
    assert score(value)["reward"] == 1


@pytest.mark.parametrize(
    "configuration",
    [
        {"type": "not_registered"},
        {"weights": [1, 1, 1]},
        {"weights": [True, 0, 0]},
        {"numeric": {"alpha": -1}},
        {"numeric": {"minimum_scale": 0}},
        {"numeric": {"relative_tolerance": float("nan")}},
        {"numeric": {"alpha": float("inf")}},
        {"unknown_option": "ignored?"},
    ],
)
def test_configuration_rejects_unsupported_or_nonfinite_values(configuration):
    with pytest.raises((ValueError, TypeError)):
        RewardConfig.from_dict(configuration)


def test_configurable_numeric_policy_and_fingerprint():
    config = RewardConfig.from_dict({"type": "execution", "numeric": {"alpha": 2.0}})
    assert config.numeric == NumericReward(alpha=2)
    assert config.fingerprint() != RewardConfig(type="execution").fingerprint()
    assert (
        Decimal("0") <= Decimal(str(score_execution(context(), config)["reward"])) <= Decimal("1")
    )
