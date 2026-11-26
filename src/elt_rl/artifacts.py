"""Atomic JSON artifacts shared by the CLI, environments and trainer."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def implementation_fingerprint() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted([*root.rglob("*.py"), *root.glob("*.json")])
    }


def valid_submission_receipt(raw: dict) -> bool:
    """Check that a submitted native episode and its direct grader receipt agree."""
    result = raw.get("execution_result")
    report = raw.get("report")
    if not isinstance(result, dict) or not isinstance(report, dict):
        return False
    official = result.get("official")
    if not isinstance(official, dict):
        return False
    score = official.get("official_score")
    success = official.get("official_success")
    reward = report.get("training_reward")
    receipt = result.get("reward_result")
    if receipt is None:
        reward_matches = reward == report.get("official_score") == score
    else:
        reward_matches = (
            isinstance(receipt, dict)
            and receipt == report.get("reward_result")
            and type(reward) in (int, float)
            and math.isfinite(reward)
            and 0 <= reward <= 1
            and reward == receipt.get("reward")
            and isinstance(receipt.get("artifact"), dict)
            and isinstance(receipt.get("config_sha256"), str)
            and len(receipt["config_sha256"]) == 64
            and all(
                type(receipt.get(name)) in (int, float)
                and math.isfinite(receipt[name])
                and 0 <= receipt[name] <= 1
                for name in (
                    "cell_accuracy",
                    "column_score",
                    "cell_score",
                    "value_score",
                )
                if name in receipt
            )
            and report.get("official_score") == score
        )
        if reward_matches and receipt.get("reward_type") == "pure_cell":
            correct, required = receipt.get("correct_cells"), receipt.get("required_cells")
            mask = receipt.get("cell_results")
            reward_matches = (
                type(correct) is int
                and type(required) is int
                and 0 <= correct <= required
                and required > 0
                and isinstance(mask, list)
                and len(mask) == required
                and all(type(cell.get("correct")) is bool for cell in mask)
                and sum(cell["correct"] for cell in mask) == correct
                and reward == correct / required == report.get("cell_accuracy")
                and report.get("correct_cells") == correct
                and report.get("required_cells") == required
                and report.get("cell_results") == mask
            )
    local_fixture = (
        receipt is not None
        and raw.get("mode") == "local_test"
        and official.get("evaluation_provider") == "local_fixture"
    )
    valid_official = (
        (
            score is None
            and success is None
            and type(official.get("fixture_success")) is bool
            and report.get("success") is official["fixture_success"]
        )
        if local_fixture
        else (
            type(score) in (float, int)
            and 0 <= score <= 1
            and type(success) is bool
            and report.get("success") is success
        )
    )
    return (
        result.get("kind") == "submission"
        and result.get("execution_success") is True
        and result.get("pipeline_valid") is True
        and isinstance(result.get("execution"), dict)
        and report.get("execution_success") is True
        and report.get("pipeline_valid") is True
        and valid_official
        and reward_matches
        and report.get("official_success") is success
    )


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_finite(value), default=str, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value
