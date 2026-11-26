from dataclasses import replace
from decimal import Decimal

import pytest

from elt_rl.grading import ReferenceError, cell_equal, compare
from elt_rl.warehouse import TableData


def gold_rows(task):
    return TableData(
        task.targets[0].columns,
        (
            ("004", "Dana", 1, 0),
            ("001", "Alice", 2, Decimal("8.25")),
            ("003", "Chen", 0, 0),
            ("002", "Bob", 1, Decimal("7.50")),
        ),
    )


def test_key_aligned_not_order_sensitive(task):
    assert compare(gold_rows(task), task.targets[0]).passed


def test_null_nan_case_and_precision():
    atol, rtol = Decimal("1e-8"), Decimal("1e-6")
    assert cell_equal(None, "", True, atol, rtol)
    assert not cell_equal(float("nan"), "3.0", True, atol, rtol)
    assert not cell_equal(None, "0", True, atol, rtol)
    assert not cell_equal("alice", "Alice", False, atol, rtol)
    assert not cell_equal("001", "1", False, atol, rtol)
    assert not cell_equal(9007199254740992, "9007199254740993", True, atol, rtol)
    assert not cell_equal(1000001, "1000000", True, atol, rtol)
    assert cell_equal("8.2500000001", "8.25", True, atol, rtol)


def test_duplicate_keys_wrong_schema_and_rows(task):
    data = gold_rows(task)
    duplicate = replace(data, rows=(data.rows[0],) * 4)
    assert compare(duplicate, task.targets[0]).score == 0
    assert compare(replace(data, rows=data.rows[:3]), task.targets[0]).score == 0
    assert compare(replace(data, columns=("x",)), task.targets[0]).score == 0
    assert compare(replace(data, truncated=True), task.targets[0]).score == 0


def test_broken_gold_is_infrastructure_error(task, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("wrong\n1\n")
    target = replace(task.targets[0], expected=path)
    with pytest.raises(ReferenceError, match="Trusted reference schema"):
        compare(gold_rows(task), target)
