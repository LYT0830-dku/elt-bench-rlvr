"""Explicit authorization for task, model and experiment size."""

import json
from pathlib import Path

from elt_rl.registry import UnsupportedCapability


def validate_experiment_scope(config, tasks):
    policy = config.authorization
    required = {"models", "tasks", "max_batches", "max_concurrency", "fresh_base_only"}
    if (
        not isinstance(policy, dict)
        or not required <= set(policy)
        or set(policy) - required - {"request_sha256"}
    ):
        raise ValueError("Remote execution requires an explicit experiment authorization")
    if config.model not in policy["models"] or any(
        task.id not in policy["tasks"] for task in tasks
    ):
        raise UnsupportedCapability("Selected task or model is outside experiment authorization")
    if config.batches > policy["max_batches"] or config.concurrency > policy["max_concurrency"]:
        raise ValueError("Experiment exceeds authorized batch count or concurrency")
    if type(policy["fresh_base_only"]) is not bool:
        raise ValueError("fresh_base_only must be explicit")
    if policy["fresh_base_only"] and config.resume_checkpoint:
        saved = json.loads(Path(config.resume_checkpoint).read_text())
        if (
            saved.get("config", {}).get("run_id") != config.run_id
            or saved.get("config", {}).get("model") != config.model
        ):
            raise ValueError("A fresh-base experiment may resume only its own confirmed checkpoint")
