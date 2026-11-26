"""Run-wide dispatch deadline and persistent per-node evaluation attempt limits."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from elt_rl.errors import InfrastructureError, IntegrityError


class RunDeadlineExceeded(InfrastructureError):
    """Stop new work while retaining in-flight accounting and allowing cleanup."""


class EvaluationAttemptsExhausted(InfrastructureError):
    """This node has consumed its predeclared evaluation retry allowance."""


def deadline_epoch(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("deadline_utc must be an ISO-8601 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid deadline_utc") from exc
    if parsed.tzinfo is None:
        raise ValueError("deadline_utc must include timezone")
    epoch = parsed.astimezone(timezone.utc).timestamp()
    if not math.isfinite(epoch):
        raise ValueError("Invalid deadline_utc")
    return epoch


@dataclass
class RunControl:
    steps: int
    deadline_utc: str | None = None
    max_evaluation_attempts: int = 2
    evaluation_attempts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        self.deadline = deadline_epoch(self.deadline_utc)
        if (
            isinstance(self.max_evaluation_attempts, bool)
            or not isinstance(self.max_evaluation_attempts, int)
            or self.max_evaluation_attempts < 1
        ):
            raise ValueError("max_evaluation_attempts must be a positive integer")
        self.evaluation_attempts = dict(self.evaluation_attempts)
        if any(
            not isinstance(k, str)
            or not k.isascii()
            or not k.isdecimal()
            or k != str(int(k))
            or not 0 <= int(k) <= self.steps
            or isinstance(v, bool)
            or not isinstance(v, int)
            or not 0 <= v <= self.max_evaluation_attempts
            for k, v in self.evaluation_attempts.items()
        ):
            raise IntegrityError("Invalid cumulative evaluation-attempt ledger")

    def check_dispatch(self):
        if self.deadline is not None and time.time() >= self.deadline:
            raise RunDeadlineExceeded("Global run deadline reached; no new work dispatched")

    def claim_evaluation(self, step):
        self.check_dispatch()
        if type(step) is not int or not 0 <= step <= self.steps:
            raise ValueError("Invalid evaluation step")
        key = str(step)
        used = self.evaluation_attempts.get(key, 0)
        if used >= self.max_evaluation_attempts:
            raise EvaluationAttemptsExhausted(
                f"Step {step} reached its {self.max_evaluation_attempts} evaluation-attempt limit"
            )
        self.evaluation_attempts[key] = used + 1

    def public(self):
        return {
            "planned_updates": self.steps,
            "deadline_utc": self.deadline_utc,
            "max_evaluation_attempts": self.max_evaluation_attempts,
            "evaluation_attempts": dict(self.evaluation_attempts),
        }
