"""Validated nested component selection, lowered into the existing training loop."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from elt_rl.model_registry import MODELS
from elt_rl.reward_registry import REWARDS
from elt_rl.rewards import RewardConfig
from elt_rl.task_scheduling import TaskSchedule
from elt_rl.warehouse_backends import WAREHOUSES


def lower_configuration(data, path, allow_remote=False):
    from elt_rl.train import TrainConfig

    allowed = {
        "format",
        "run_id",
        "output",
        "task",
        "tasks",
        "model",
        "warehouse",
        "reward",
        "training",
        "evaluation",
        "limits",
        "authorization",
        "resume_checkpoint",
    }
    if set(data) - allowed or data.get("format") != "elt_framework_v1":
        raise ValueError("Unknown/unsupported framework configuration field")
    base = Path(path).resolve().parent

    def resolve(value):
        return str((base / value).resolve())

    entries = data.get("tasks")
    if entries is not None and "task" in data:
        raise ValueError("Specify task or tasks, not both")
    entries = entries if entries is not None else [data["task"]]
    groups = []
    for entry in entries:
        if set(entry) != {"id", "spec"}:
            raise ValueError("Task selection requires id and spec")
        groups.append({"task_id": entry["id"], "specification": resolve(entry["spec"])})
    evaluation = dict(data.get("evaluation", {}))
    schedule = TaskSchedule(groups, evaluation.pop("task", None))
    task_collection = schedule.load()
    model = dict(data["model"])
    if set(model) - {"id", "renderer", "renderer_effort", "context_tokens"}:
        raise ValueError("Unknown model selection field")
    adapter = MODELS.get(model["id"])
    warehouse = data["warehouse"]
    if set(warehouse) != {"backend"}:
        raise ValueError("Unknown warehouse selection field")
    WAREHOUSES.get(warehouse["backend"])
    reward = RewardConfig.from_dict(data.get("reward")).configuration()
    REWARDS.get(reward["type"])
    training = dict(data.get("training", {}))
    if set(training) - {
        "batches",
        "group_size",
        "learning_rate",
        "rank",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "concurrency",
    }:
        raise ValueError("Unknown/misplaced training parameter")
    limits = dict(data.get("limits", {}))
    allowed_limits = {
        "max_turns",
        "max_tokens",
        "context_tokens",
        "max_episode_tokens",
        "max_sampling_tokens",
        "max_training_tokens",
        "max_seconds",
        "sampling_timeout_seconds",
        "optimization_timeout_seconds",
        "checkpoint_timeout_seconds",
        "max_evaluation_attempts",
        "deadline_utc",
    }
    if set(limits) - allowed_limits:
        raise ValueError("Unknown/misplaced execution budget")
    evaluation_limits = dict(evaluation.pop("limits", {}))
    options = {
        "task": str(task_collection.primary.execution_task.manifest),
        "task_spec_path": groups[0]["specification"],
        "task_groups": tuple(groups),
        "evaluation_task": schedule.evaluation_task,
        "model": model["id"],
        "renderer": adapter.renderer_name(model.get("renderer")),
        "renderer_effort": model.get(
            "renderer_effort", 0.0 if adapter.configuration["effort_conditioning"] else None
        ),
        "context_tokens": model.get("context_tokens", adapter.configuration["context_limit"]),
        "destination": warehouse["backend"],
        "reward": reward,
        "run_id": data["run_id"],
        "output": resolve(data["output"]),
        "schedule": "batches_v1",
        "allow_remote": allow_remote,
        "evaluation_limits": evaluation_limits,
        "authorization": dict(data.get("authorization", {})),
    }
    options.update(training)
    options.setdefault("concurrency", min(8, options.get("group_size", 8)))
    evaluation_aliases = {
        "enabled": "evaluation_enabled",
        "interval": "evaluation_interval",
        "rollouts": "eval_episodes",
        "final_rollouts": "final_eval_episodes",
        "seed": "final_seed",
        "temperature": "evaluation_temperature",
    }
    for name, value in evaluation.items():
        key = evaluation_aliases.get(name)
        if key is None:
            raise ValueError("Unknown evaluation option: " + name)
        options[key] = value
    options.update(limits)
    if data.get("resume_checkpoint"):
        options["resume_checkpoint"] = resolve(data["resume_checkpoint"])
    unknown = set(options) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError("Unknown training/limits configuration: " + ", ".join(sorted(unknown)))
    result = TrainConfig(**options)
    result.validate()
    return result
