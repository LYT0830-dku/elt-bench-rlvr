"""Deterministic, host-only execution rewards, independent of official evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from elt_rl.errors import InfrastructureError


class RewardInfrastructureError(InfrastructureError):
    """Broken trusted data or unknown execution state cannot become policy zero."""


@dataclass(frozen=True)
class NumericReward:
    alpha: float = 5.0
    relative_tolerance: float = 0.01
    minimum_scale: float = 1.0
    absolute_tolerance: float = 0.0
    zero_absolute_tolerance: float = 0.0

    def validate(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Nonfinite/invalid numeric reward parameter: " + name)
            if value < 0 or (name in {"alpha", "minimum_scale"} and value == 0):
                raise ValueError("Invalid numeric reward range: " + name)
        if self.relative_tolerance > 1:
            raise ValueError("relative_tolerance must be in [0, 1]")


@dataclass(frozen=True)
class RewardConfig:
    type: str = "pure_cell"
    numeric: NumericReward = field(default_factory=NumericReward)
    weights: tuple[float, float, float] | None = None
    version: str | None = None

    def __post_init__(self):
        if self.version is None:
            object.__setattr__(self, "version", REWARD_VERSIONS.get(self.type))

    @classmethod
    def from_dict(cls, value=None):
        value = dict(value or {})
        if "type" not in value and value.get("version") == "execution_reward_v1":
            value["type"] = "execution"
        unknown = set(value) - {"type", "numeric", "weights", "version"}
        if unknown:
            raise ValueError("Unknown reward configuration: " + ", ".join(sorted(unknown)))
        if value.get("type", "pure_cell") == "pure_cell" and (
            "weights" in value or "numeric" in value
        ):
            raise ValueError("Pure cell accuracy has no mixture weights or proximity parameters")
        numeric = NumericReward(**value.pop("numeric", {}))
        weights = value.pop("weights", None)
        if isinstance(weights, dict):
            if set(weights) != {"column", "cell", "value"}:
                raise ValueError("Named weights require column, cell and value")
            weights = [weights[name] for name in ("column", "cell", "value")]
        result = cls(
            numeric=numeric, weights=tuple(weights) if weights is not None else None, **value
        )
        result.validate()
        return result

    def validate(self):
        if self.type not in REWARD_VERSIONS:
            raise ValueError("Unregistered reward type: " + self.type)
        self.numeric.validate()
        if self.version != REWARD_VERSIONS[self.type]:
            raise ValueError("Unsupported reward version")
        if self.type == "pure_cell":
            if self.weights is not None or self.numeric != NumericReward():
                raise ValueError(
                    "Pure cell accuracy has no mixture weights or proximity parameters"
                )
            return
        weights = self.weights if self.weights is not None else HISTORICAL_WEIGHTS[self.type]
        if (
            len(weights) != 3
            or any(
                isinstance(w, bool)
                or not isinstance(w, (int, float))
                or not math.isfinite(w)
                or w < 0
                for w in weights
            )
            or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("Reward weights must be three finite nonnegative values summing to 1")

    def configuration(self):
        if self.type == "pure_cell":
            return {"type": self.type, "version": self.version}
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.configuration(), sort_keys=True).encode()).hexdigest()


DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)
HISTORICAL_WEIGHTS = {
    "execution": DEFAULT_WEIGHTS,
    "column_execution": (1.0, 0.0, 0.0),
    "column_cell_execution": (0.5, 0.5, 0.0),
    "hierarchical_execution": DEFAULT_WEIGHTS,
}
REWARD_VERSIONS = {
    "pure_cell": "pure_cell_accuracy_v1",
    "execution": "execution_reward_v1",
    **{name: "hierarchical_execution_v1" for name in HISTORICAL_WEIGHTS if name != "execution"},
}
REWARD_METRICS = (
    "cell_accuracy",
    "correct_cells",
    "required_cells",
    "cell_score",
    "column_score",
    "value_score",
)


@dataclass(frozen=True)
class RewardField:
    name: str
    kind: str = "string"
    role: str = "business"
    nullable: bool = False
    minimum: str | None = None
    maximum: str | None = None
    comparator: str = "exact"
    cell_rtol: float = 0.01
    cell_atol: float = 0.0
    numeric: NumericReward | None = None

    def validate(self):
        if (
            not self.name
            or self.kind not in {"string", "integer", "decimal", "boolean", "date", "datetime"}
            or self.role not in {"business", "identifier", "categorical"}
        ):
            raise ValueError("Invalid reward field specification")
        if self.comparator not in {"exact", "relative", "official_elt_relative_v1"}:
            raise ValueError("Unknown field comparator")
        for value in (self.cell_rtol, self.cell_atol):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Invalid cell tolerance")
        if self.numeric:
            self.numeric.validate()
        if (
            self.minimum is not None
            and self.maximum is not None
            and Decimal(self.minimum) > Decimal(self.maximum)
        ):
            raise ValueError("Invalid field domain")


@dataclass(frozen=True)
class RewardTarget:
    name: str
    fields: tuple[RewardField, ...]
    keys: tuple[str, ...]

    def validate(self):
        if not self.name or not self.fields or not self.keys:
            raise ValueError(
                "Fine-grained reward requires a target and verified row-alignment keys"
            )
        names = [field.name.casefold() for field in self.fields]
        if (
            len(names) != len(set(names))
            or len(self.keys) != len(set(k.casefold() for k in self.keys))
            or not set(k.casefold() for k in self.keys) <= set(names)
        ):
            raise ValueError("Invalid target fields/keys")
        for specification in self.fields:
            specification.validate()


@dataclass(frozen=True)
class RewardTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class RewardContext:
    task_id: str
    targets: tuple[RewardTarget, ...]
    predictions: dict[str, RewardTable]
    references: dict[str, RewardTable]
    official_columns: dict[str, dict[str, bool]]
    stage1_passed: bool | None
    submission_valid: bool | None
    infrastructure_error: str | None = None


def _parsed(value, specification):
    if value is None:
        if specification.nullable:
            return None
        raise ValueError("NULL is not allowed")
    if specification.kind in {"integer", "decimal"}:
        if isinstance(value, bool):
            raise ValueError("Boolean is not a numeric business value")
        number = Decimal(str(value))
        if not number.is_finite() or abs(number.adjusted()) > 10000:
            raise ValueError("Nonfinite/out-of-range numeric value")
        if specification.kind == "integer" and number != number.to_integral_value():
            raise ValueError("Noninteger count/identifier")
        if specification.minimum is not None and number < Decimal(specification.minimum):
            raise ValueError("Below field domain")
        if specification.maximum is not None and number > Decimal(specification.maximum):
            raise ValueError("Above field domain")
        return number
    if specification.kind == "date":
        if isinstance(value, datetime):
            raise ValueError("Datetime is not a date")
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    if specification.kind == "datetime":
        return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if specification.kind == "boolean":
        if value is True or str(value).casefold() in {"true", "1"}:
            return True
        if value is False or str(value).casefold() in {"false", "0"}:
            return False
        raise ValueError("Invalid boolean")
    if not isinstance(value, str):
        raise ValueError("Invalid string")
    return value


def _equal(actual, expected, specification, *, key=False):
    if actual is None or expected is None:
        return actual is None and expected is None
    if (
        key
        or specification.role != "business"
        or specification.kind not in {"integer", "decimal"}
        or specification.comparator == "exact"
    ):
        return actual == expected
    relative, absolute = (
        Decimal(str(specification.cell_rtol)),
        Decimal(str(specification.cell_atol)),
    )
    if specification.comparator == "official_elt_relative_v1":
        # Preserve the original comparator's asymmetric negative-value behavior.
        return expected * (1 - relative) <= actual <= expected * (1 + relative)
    return abs(actual - expected) <= absolute + relative * abs(expected)


def _value(actual, expected, correct, specification, config, *, key=False):
    if correct:
        return 1.0, None
    if (
        actual is None
        or expected is None
        or key
        or specification.role != "business"
        or specification.kind not in {"integer", "decimal"}
    ):
        return 0.0, None
    numeric = specification.numeric or config.numeric
    tolerance = (
        Decimal(str(numeric.zero_absolute_tolerance))
        if expected == 0
        else max(
            Decimal(str(numeric.absolute_tolerance)),
            abs(expected) * Decimal(str(numeric.relative_tolerance)),
        )
    )
    error = max(Decimal(0), abs(actual - expected) - tolerance) / max(
        abs(expected), Decimal(str(numeric.minimum_scale))
    )
    exponent = Decimal(str(numeric.alpha)) * error
    score = 0.0 if exponent > 745 else math.exp(-float(exponent))
    return score, str(error)


def _rows(table, target, *, reference):
    label = "reference" if reference else "prediction"
    columns = [column.casefold() for column in table.columns]
    if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in table.rows):
        raise ValueError("Malformed " + label + " table")
    fields = {specification.name.casefold(): specification for specification in target.fields}
    if reference and not set(fields) <= set(columns):
        raise ValueError("Reference is missing required columns")
    if not set(k.casefold() for k in target.keys) <= set(columns):
        raise ValueError("Missing primary-key column")
    result = {}
    for row in table.rows:
        values = dict(zip(columns, row, strict=True))
        key = tuple(_parsed(values[k.casefold()], fields[k.casefold()]) for k in target.keys)
        if any(value is None for value in key) or key in result:
            raise ValueError("Null/duplicate primary key")
        if reference:
            for name, specification in fields.items():
                _parsed(values[name], specification)
        result[key] = values
    if reference and not result:
        raise ValueError("Empty trusted reference requires an explicit alternate alignment policy")
    return result


def score_execution(context: RewardContext, config: RewardConfig | None = None) -> dict:
    config = config or RewardConfig()
    config.validate()
    if config.type == "pure_cell":
        return _score_pure_cell(context, config)
    return _score_historical(context, config)


def _score_pure_cell(context, config):
    if (
        context.infrastructure_error
        or type(context.stage1_passed) is not bool
        or type(context.submission_valid) is not bool
    ):
        raise RewardInfrastructureError(
            context.infrastructure_error or "Unknown execution gate state"
        )
    names = [target.name for target in context.targets]
    if not names or len(names) != len(set(names)):
        raise RewardInfrastructureError("Invalid trusted target specification")
    gate = (
        "invalid_submission"
        if not context.submission_valid
        else "loading_failed"
        if not context.stage1_passed
        else None
    )
    targets = {}
    for target in context.targets:
        try:
            target.validate()
            reference = context.references.get(target.name)
            if reference is None:
                raise ValueError("Missing trusted reference")
            expected = _rows(reference, target, reference=True)
        except (ValueError, InvalidOperation, ArithmeticError) as exc:
            raise RewardInfrastructureError("Invalid trusted reference: " + target.name) from exc
        actual, error = {}, gate
        prediction = context.predictions.get(target.name)
        if gate is None:
            try:
                if prediction is None:
                    error = "Missing target table"
                else:
                    actual = _rows(prediction, target, reference=False)
                    if actual.keys() - expected.keys():
                        raise ValueError("Extra primary-key rows invalidate this target")
            except (ValueError, InvalidOperation, ArithmeticError) as exc:
                actual, error = {}, str(exc)
        cells, columns = [], {}
        keys = {key.casefold() for key in target.keys}
        for specification in target.fields:
            name = specification.name.casefold()
            selected = []
            for key, expected_row in expected.items():
                actual_row = actual.get(key, {})
                missing = name not in actual_row
                correct, invalid = False, error
                if error is None:
                    try:
                        if missing:
                            raise ValueError("Missing required cell")
                        correct = _equal(
                            _parsed(actual_row[name], specification),
                            _parsed(expected_row[name], specification),
                            specification,
                            key=name in keys,
                        )
                    except (ValueError, InvalidOperation, ArithmeticError) as exc:
                        invalid = str(exc)
                cell = {
                    "key": [str(component) for component in key],
                    "column": specification.name,
                    "prediction": actual_row.get(name),
                    "reference": expected_row[name],
                    "missing": missing,
                    "correct": bool(correct),
                    "value": int(correct),
                    "invalid": invalid,
                }
                cells.append(cell)
                selected.append(cell)
            count = sum(cell["correct"] for cell in selected)
            columns[specification.name] = {
                "correct_cells": count,
                "required_cells": len(expected),
                "cell_accuracy": count / len(expected),
            }
        correct_count = sum(cell["correct"] for cell in cells)
        targets[target.name] = {
            "correct_cells": correct_count,
            "required_cells": len(cells),
            "cell_accuracy": correct_count / len(cells),
            "cell_score": correct_count / len(cells),
            "reward": correct_count / len(cells),
            "columns": columns,
            "cells": cells,
            "error": error,
        }
    denominator = sum(target["required_cells"] for target in targets.values())
    numerator = sum(target["correct_cells"] for target in targets.values())
    return {
        "reward": numerator / denominator,
        "cell_accuracy": numerator / denominator,
        "cell_score": numerator / denominator,
        "correct_cells": numerator,
        "required_cells": denominator,
        "aggregation": "micro",
        "reward_type": config.type,
        "version": config.version,
        "config": config.configuration(),
        "config_sha256": config.fingerprint(),
        "targets": targets,
        "gate_error": gate,
        "cell_results": [
            {
                "target": name,
                "key": cell["key"],
                "column": cell["column"],
                "correct": cell["correct"],
            }
            for name, target in targets.items()
            for cell in target["cells"]
        ],
    }


def _score_historical(context: RewardContext, config: RewardConfig) -> dict:
    if (
        context.infrastructure_error
        or type(context.stage1_passed) is not bool
        or type(context.submission_valid) is not bool
    ):
        raise RewardInfrastructureError(
            context.infrastructure_error or "Unknown execution gate state"
        )
    names = [target.name for target in context.targets]
    if not names or len(names) != len(set(names)):
        raise RewardInfrastructureError("Invalid trusted target specification")
    if not context.stage1_passed or not context.submission_valid:
        return _result(
            config, {}, "loading_failed" if not context.stage1_passed else "invalid_submission"
        )
    targets = {}
    for target in context.targets:
        target.validate()
        reference = context.references.get(target.name)
        if reference is None:
            raise RewardInfrastructureError("Missing trusted reference: " + target.name)
        try:
            expected = _rows(reference, target, reference=True)
        except (ValueError, InvalidOperation, ArithmeticError) as exc:
            raise RewardInfrastructureError("Invalid trusted reference: " + target.name) from exc
        prediction = context.predictions.get(target.name)
        try:
            if prediction is None:
                raise ValueError("Missing target table")
            actual = _rows(prediction, target, reference=False)
            if actual.keys() != expected.keys():
                raise ValueError("Missing/extra primary-key rows")
        except (ValueError, InvalidOperation, ArithmeticError) as exc:
            targets[target.name] = {
                "column_score": 0.0,
                "cell_score": 0.0,
                "value_score": 0.0,
                "error": str(exc),
                "columns": {},
                "cells": [],
                "required_cells": len(expected) * len(target.fields),
            }
            continue
        official = context.official_columns.get(target.name)
        if (
            official is None
            or set(k.casefold() for k in official) != {f.name.casefold() for f in target.fields}
            or any(type(value) is not bool for value in official.values())
        ):
            raise RewardInfrastructureError("Incomplete official column receipt: " + target.name)
        official = {name.casefold(): value for name, value in official.items()}
        columns, cells = {}, []
        keys = {key.casefold() for key in target.keys}
        for specification in target.fields:
            name = specification.name.casefold()
            field_cells = []
            for key, expected_row in expected.items():
                actual_row = actual[key]
                missing = name not in actual_row
                correct, value, normalized_error, invalid = False, 0.0, None, None
                expected_value = _parsed(expected_row[name], specification)
                try:
                    if missing:
                        raise ValueError("Missing required cell")
                    actual_value = _parsed(actual_row[name], specification)
                    correct = _equal(actual_value, expected_value, specification, key=name in keys)
                    value, normalized_error = _value(
                        actual_value,
                        expected_value,
                        correct,
                        specification,
                        config,
                        key=name in keys,
                    )
                except (ValueError, InvalidOperation, ArithmeticError) as exc:
                    invalid = str(exc)
                cell = {
                    "key": [str(component) for component in key],
                    "column": specification.name,
                    "prediction": actual_row.get(name),
                    "reference": expected_row[name],
                    "missing": missing,
                    "correct": correct,
                    "value": value,
                    "normalized_numeric_error": normalized_error,
                    "invalid": invalid,
                    "custom_value_comparator_difference": (value == 1.0) != correct,
                }
                cells.append(cell)
                field_cells.append(cell)
            present = all(name in row for row in actual.values())
            if official[name] and not present:
                raise RewardInfrastructureError("Official receipt claims a missing column passed")
            columns[specification.name] = {
                "official_match": official[name],
                "cell_score": sum(cell["correct"] for cell in field_cells) / len(expected),
                "value_score": sum(cell["value"] for cell in field_cells) / len(expected),
            }
        targets[target.name] = {
            "column_score": sum(value["official_match"] for value in columns.values())
            / len(target.fields),
            "cell_score": sum(cell["correct"] for cell in cells) / len(cells),
            "value_score": sum(cell["value"] for cell in cells) / len(cells),
            "columns": columns,
            "cells": cells,
            "required_cells": len(cells),
            "error": None,
        }
    return _result(config, targets, None)


def _result(config, targets, gate_error):
    components = {
        name: sum(target[name] for target in targets.values()) / len(targets) if targets else 0.0
        for name in ("column_score", "cell_score", "value_score")
    }
    weights = config.weights if config.weights is not None else HISTORICAL_WEIGHTS[config.type]
    selected = sum(
        weight * components[key] for weight, key in zip(weights, components, strict=True)
    )
    for target in targets.values():
        target["reward"] = sum(
            weight * target[key] for weight, key in zip(weights, components, strict=True)
        )
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in [selected, *components.values()]
    ):
        raise RewardInfrastructureError("Nonfinite/out-of-range execution reward")
    return {
        "reward": selected,
        "reward_type": config.type,
        "version": config.version,
        "config": asdict(config),
        "config_sha256": config.fingerprint(),
        **components,
        "targets": targets,
        "gate_error": gate_error,
    }
