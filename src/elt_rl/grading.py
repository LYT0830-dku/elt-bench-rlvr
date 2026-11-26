"""Key-aligned result comparison for local DuckDB fixtures."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation

from elt_rl.task import Target
from elt_rl.warehouse import TableData


class ReferenceError(RuntimeError):
    """A broken trusted fixture must abort a run, never become a policy reward."""


def number(value) -> Decimal | None:
    try:
        n = Decimal(str(value))
        return n if n.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def null(value) -> bool:
    return value is None or value == ""


def cell_equal(actual, expected, numeric: bool, atol: Decimal, rtol: Decimal) -> bool:
    if null(actual) or null(expected):
        return null(actual) and null(expected)
    if not numeric:
        return str(actual) == str(expected)
    a, b = number(actual), number(expected)
    if a is None or b is None:
        return False
    if b == b.to_integral_value():
        return a == b  # Counts and IDs do not get a relative-error allowance.
    return abs(a - b) <= atol + rtol * abs(b)


@dataclass(frozen=True)
class Grade:
    score: float
    passed: bool
    columns: dict[str, bool]
    reason: str

    def public(self):
        return asdict(self)


def compare(actual: TableData, target: Target) -> Grade:
    try:
        with target.expected.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            expected_columns = tuple(c.lower() for c in reader.fieldnames or [])
            expected = [{k.lower(): v for k, v in r.items()} for r in reader]
    except (OSError, ValueError, AttributeError) as exc:
        raise ReferenceError("Cannot read trusted reference") from exc
    if set(expected_columns) != set(target.columns):
        raise ReferenceError("Trusted reference schema does not match task manifest")
    failed = dict.fromkeys(target.columns, False)
    if actual.truncated:
        return Grade(0, False, failed, "Output exceeds grading row budget")
    if len(actual.columns) != len(set(actual.columns)) or set(actual.columns) != set(
        target.columns
    ):
        return Grade(0, False, failed, "Output schema mismatch")
    if len(actual.rows) != len(expected):
        return Grade(0, False, failed, "Output row count mismatch")
    actual_rows = [dict(zip(actual.columns, row, strict=True)) for row in actual.rows]

    def key(row):
        # Explicit numeric keys preserve 64-bit integer precision using Decimal.
        values = []
        for c in target.keys:
            value = row[c]
            if null(value):
                raise ValueError("Null key")
            if c in target.numeric:
                value = number(value)
                if value is None:
                    raise ValueError("Non-finite numeric key")
            else:
                value = str(value)
            values.append(value)
        return tuple(values)

    try:
        gold = {key(row): row for row in expected}
    except ValueError as exc:
        raise ReferenceError("Trusted reference has invalid keys") from exc
    try:
        out = {key(row): row for row in actual_rows}
    except ValueError:
        return Grade(0, False, failed, "Invalid keys")
    if len(gold) != len(expected):
        raise ReferenceError("Trusted reference has duplicate keys")
    if len(out) != len(actual_rows) or gold.keys() != out.keys():
        return Grade(0, False, failed, "Duplicate or incorrect keys")
    matches = {
        c: all(
            cell_equal(
                out[k][c],
                gold[k][c],
                c in target.numeric,
                Decimal(target.atol),
                Decimal(target.rtol),
            )
            for k in gold
        )
        for c in target.columns
    }
    score = sum(matches.values()) / len(matches)
    assert math.isfinite(score)
    return Grade(
        score,
        all(matches.values()),
        matches,
        "Matched" if all(matches.values()) else "Column values differ",
    )
