"""Bounded on-policy training and paired evaluation using real Cookbook trajectories."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from elt_rl.artifacts import implementation_fingerprint, write_json
from elt_rl.errors import TokenLimitExceeded
from elt_rl.lineage import evidence_ref, verify_lineage
from elt_rl.model_registry import DEFAULT_MODEL
from elt_rl.rewards import REWARD_METRICS, RewardConfig


@dataclass(frozen=True)
class TrainConfig:
    task: str
    output: str
    model: str = DEFAULT_MODEL
    renderer: str | None = "tml_v0"
    renderer_effort: float | None = 0.0
    action_protocol: str = "native_tools_v1"
    destination: str = "snowflake"
    batches: int = 16
    group_size: int = 8
    max_turns: int = 200
    max_tokens: int = 49152
    context_tokens: int = 64000
    learning_rate: float = 1e-5
    rank: int = 16
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0
    max_episode_tokens: int = 262144
    max_sampling_tokens: int = 1280000000
    max_training_tokens: int = 20000000
    max_seconds: float | None = None
    sampling_timeout_seconds: float = 180
    eval_episodes: int = 10
    resume_checkpoint: str | None = None
    allow_remote: bool = False
    optimization_timeout_seconds: float = 900
    checkpoint_timeout_seconds: float = 900
    concurrency: int = 8
    max_evaluation_attempts: int = 1
    deadline_utc: str | None = None
    schedule: str = "batches_v1"
    evaluation_interval: int = 4
    evaluation_enabled: bool = True
    final_eval_episodes: int = 20
    final_seed: int = 200000000
    run_id: str | None = None
    task_spec_path: str | None = None
    task_groups: tuple = ()
    evaluation_task: str | None = None
    reward: dict | None = None
    evaluation_temperature: float | None = None
    evaluation_limits: dict = field(default_factory=dict)
    authorization: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "reward", RewardConfig.from_dict(self.reward).configuration())

    def validate(self):
        from elt_rl.control import deadline_epoch
        from elt_rl.model_registry import MODELS
        from elt_rl.rewards import RewardConfig

        if self.schedule != "batches_v1" or self.action_protocol != "native_tools_v1":
            raise ValueError("Use the structured tool protocol and batch training schedule")
        if type(self.evaluation_enabled) is not bool:
            raise ValueError("evaluation_enabled must be a boolean")
        for name in (
            "batches",
            "group_size",
            "max_turns",
            "max_tokens",
            "context_tokens",
            "rank",
            "max_episode_tokens",
            "max_sampling_tokens",
            "max_training_tokens",
            "concurrency",
            "max_evaluation_attempts",
            "evaluation_interval",
            "eval_episodes",
            "final_eval_episodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.group_size < 2 or self.concurrency > self.group_size:
            raise ValueError(
                "Groups require at least two rollouts and concurrency at most group size"
            )
        if self.max_tokens >= self.context_tokens:
            raise ValueError("Completion limit must leave room for prompt tokens")
        for name in (
            "learning_rate",
            "sampling_timeout_seconds",
            "optimization_timeout_seconds",
            "checkpoint_timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("temperature", "top_p", "evaluation_temperature"):
            value = getattr(self, name)
            if value is None and name == "evaluation_temperature":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if (
            not 0 < self.top_p <= 1
            or type(self.top_k) is not int
            or (self.top_k != -1 and self.top_k < 1)
        ):
            raise ValueError("Invalid sampling top_p or top_k")
        for name in ("seed", "final_seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**31:
                raise ValueError(name + " must be an integer in [0, 2**31)")
        if self.renderer_effort is not None and (
            type(self.renderer_effort) not in (int, float)
            or not math.isfinite(self.renderer_effort)
            or not 0 <= self.renderer_effort < 1
        ):
            raise ValueError("renderer_effort must be finite and in [0, 1)")
        if self.temperature <= 0 or (
            self.evaluation_temperature is not None and self.evaluation_temperature <= 0
        ):
            raise ValueError("Sampling temperatures must be positive")
        if self.max_seconds is not None and (
            type(self.max_seconds) not in (int, float)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("Episode time limit must be finite and positive")
        if set(self.evaluation_limits) - {
            "max_turns",
            "max_tokens",
            "max_episode_tokens",
            "max_seconds",
        }:
            raise ValueError("Unknown evaluation execution limit")
        for name, value in self.evaluation_limits.items():
            if name == "max_seconds" and value is None:
                continue
            if (
                type(value) not in ((int, float) if name == "max_seconds" else (int,))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid evaluation limit: " + name)
        if not self.task_spec_path:
            raise ValueError("Training requires a verified task specification")
        if not self.run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id):
            raise ValueError("run_id must be a plain independent experiment identifier")
        deadline_epoch(self.deadline_utc)
        MODELS.get(self.model).validate(
            renderer=self.renderer,
            context_tokens=self.context_tokens,
            effort=self.renderer_effort,
            training=True,
        )
        RewardConfig.from_dict(self.reward)


def load_configured_tasks(config):
    from elt_rl.task import Task
    from elt_rl.task_scheduling import TaskSchedule
    from elt_rl.task_spec import load_task_spec

    if config.task_groups:
        return TaskSchedule(config.task_groups, config.evaluation_task).load()
    if config.task_spec_path:
        return load_task_spec(config.task_spec_path)
    return Task.load(config.task)


def configured_reward_version(config):
    return RewardConfig.from_dict(config.reward).version


def preflight(config: TrainConfig, *, training=True):
    """Validate the task, runtime settings and resource limits before creating a service client."""
    config.validate()
    if training:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            raise RuntimeError("Install the report extra before formal training")
    from elt_rl.benchmark import task_material
    from elt_rl.experiment_policy import validate_experiment_scope
    from elt_rl.source_adapters import SOURCES

    task = load_configured_tasks(config)
    tasks = (
        [task.registry.get(name) for name in task.registry.names()]
        if config.task_groups
        else [task]
    )
    if not config.task_spec_path and (task.mode != "native" or config.destination != "snowflake"):
        raise ValueError(
            "The only production training path is original native ELT-Bench on Snowflake"
        )
    for selected in tasks:
        if selected.mode == "native":
            task_material(Path(selected.runtime["benchmark_repo"]), selected.name)
        if config.task_spec_path:
            from elt_rl.evaluation import EVALUATORS
            from elt_rl.warehouse_backends import WAREHOUSES

            selected.verify()
            SOURCES.validate(selected.sources)
            WAREHOUSES.get(config.destination)
            if (
                config.destination not in selected.evaluation.warehouse_backends
                or config.destination
                not in EVALUATORS.get(selected.evaluation.provider).warehouse_backends
            ):
                raise ValueError("Task does not support the selected warehouse/evaluator")
    validate_experiment_scope(config, tasks)
    if not config.allow_remote:
        raise RuntimeError("Remote model and warehouse execution require explicit authorization")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is missing; no remote requests were made")
    return task


def fingerprint(config: TrainConfig, task) -> dict:
    from elt_rl.history import (
        HISTORY_COMPACTION_RATIO,
        MODEL_HISTORY_POLICY_VERSION,
        REASONING_RESPONSE_RESERVE,
    )
    from elt_rl.model_registry import MODELS

    result = {
        "task": task.fingerprint(),
        "reward_configuration": config.reward,
        "model_adapter": MODELS.get(config.model).fingerprint(),
        "reward_version": configured_reward_version(config),
        "action_protocol": config.action_protocol,
        "renderer_conditioning": {"renderer": config.renderer, "effort": config.renderer_effort},
        "sampling_protocol": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
        "model_history": {
            "policy_version": MODEL_HISTORY_POLICY_VERSION,
            "compaction_target_fraction": list(HISTORY_COMPACTION_RATIO),
            "response_headroom_tokens": min(
                REASONING_RESPONSE_RESERVE, (config.context_tokens - 1) // 4
            ),
        },
        "implementation_sha256": implementation_fingerprint(),
        "training_protocol": {
            key: getattr(config, key)
            for key in [
                "batches",
                "group_size",
                "concurrency",
                "learning_rate",
                "rank",
                "seed",
                "max_sampling_tokens",
                "max_training_tokens",
                "max_evaluation_attempts",
                "schedule",
                "evaluation_interval",
                "final_eval_episodes",
                "final_seed",
                "task_groups",
                "evaluation_task",
            ]
        },
        "evaluation_protocol": {
            "enabled": config.evaluation_enabled,
            "n": config.eval_episodes,
            "seed": (config.seed + 100_000) % (2**31),
            "temperature": config.evaluation_temperature
            if config.evaluation_temperature is not None
            else config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "context_tokens": config.context_tokens,
            "max_episode_tokens": config.max_episode_tokens,
            "max_turns": config.max_turns,
            "max_seconds": config.max_seconds,
            "overrides": config.evaluation_limits,
        },
    }
    # Normalize tuples before persistence; recovery still requires exact equality.
    return json.loads(json.dumps(result, sort_keys=True))


def checked_checkpoint(
    path: str | None, config: TrainConfig, inputs: dict, *, for_evaluation: bool = False
) -> dict | None:
    if path is None:
        return None
    from elt_rl.resume import validate_resume_owner

    validate_resume_owner(path)
    saved = json.loads(Path(path).read_text())
    if saved.get("fingerprint") != inputs:
        raise ValueError("Checkpoint task/source fingerprints differ from the current task")
    names = ["model", "rank", "renderer", "renderer_effort", "action_protocol", "destination"]
    if not for_evaluation:
        names += ["temperature", "top_p", "top_k", "seed"]
    for name in names:
        if saved.get("config", {}).get(name) != getattr(config, name):
            raise ValueError(f"Checkpoint configuration differs: {name}")
    if not isinstance(saved.get("updates"), int) or saved["updates"] < 0:
        raise ValueError("Checkpoint has no valid update count")
    if not saved.get("path") or not saved.get("sampler_path"):
        raise ValueError("Checkpoint needs both training-state and sampler paths")
    verify_lineage(saved)
    for journal in Path(path).parent.glob("optimization-*.json"):
        record = json.loads(journal.read_text())
        if record.get("status") in {
            "outcome_unknown",
            "forward_backward_dispatching",
            "optimizer_dispatching",
            "optimizer_service_confirmed",
            "effective_update_confirmed_checkpoint_pending",
            "service_confirmed_without_nonzero_gradient",
        }:
            raise ValueError(
                "A previous optimization requires reconciliation; do not resume an earlier checkpoint"
            )
    for journal in Path(path).parent.glob("checkpoint-dispatch-*.json"):
        if json.loads(journal.read_text()).get("status") != "confirmed":
            raise ValueError("A checkpoint save requires reconciliation before resume")
    return saved


def prepare_data(data) -> tuple[list, dict]:
    """Keep Cookbook masks locally, send exactly the three supported RL tensors."""
    required = {"target_tokens", "logprobs", "advantages"}
    cleaned, input_tokens, action_tokens = [], 0, 0
    for datum in data:
        values = datum.loss_fn_inputs
        if not (required | {"mask"}) <= values.keys() or set(values) - required - {"mask"}:
            raise ValueError("Invalid importance-sampling tensor fields")
        arrays = {name: list(values[name].data) for name in required}
        length = datum.model_input.length
        if not length or any(len(array) != length for array in arrays.values()):
            raise ValueError("Misaligned RL input/target/logprob/advantage tensors")
        for name in ("logprobs", "advantages"):
            if not all(math.isfinite(float(v)) for v in arrays[name]):
                raise ValueError(f"Non-finite training {name}")
        if "mask" in values:
            mask = list(values["mask"].data)
            if len(mask) != length or any(m not in (0, 1) for m in mask):
                raise ValueError("Invalid action-token mask")
            if any(a != 0 and m == 0 for a, m in zip(arrays["advantages"], mask, strict=True)):
                raise ValueError("Observation tokens carry nonzero advantages")
        action_tokens += sum(float(v) != 0 for v in arrays["advantages"])
        input_tokens += length
        cleaned.append(replace(datum, loss_fn_inputs={k: values[k] for k in required}))
    if not cleaned or not action_tokens:
        raise ValueError("No nonzero-advantage action tokens available for training")
    return cleaned, {
        "training_input_tokens": input_tokens,
        "nonzero_advantage_tokens": action_tokens,
        "training_datums": len(cleaned),
        "max_training_sequence_tokens": max(datum.model_input.length for datum in cleaned),
    }


def versions() -> dict:
    result = {}
    for name in ("tinker", "tinker-cookbook", "transformers", "sqlglot", "duckdb"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def save_training_payload(path, data, prepared, metadata=None, episodes=None):
    """Persist exact local tensors and masks before dispatch, without state URIs."""

    def spans(values):
        return (
            None
            if values is None
            else [{"type": type(value).__name__, **asdict(value)} for value in values]
        )

    def tensor(value):
        values = value.data
        return {
            "data": values.tolist() if hasattr(values, "tolist") else list(values),
            "dtype": value.dtype,
            "shape": list(value.shape),
        }

    rows = []
    if metadata is not None and len(metadata) != len(data):
        raise ValueError("Cookbook datum-to-trajectory metadata length mismatch")
    for index, (raw, sent) in enumerate(zip(data, prepared, strict=True)):
        trajectory = None
        if metadata is not None:
            meta = metadata[index]
            if meta.get("group_idx") != 0 or not isinstance(meta.get("traj_idx"), int):
                raise ValueError("Invalid Cookbook trajectory coordinates")
            episode = episodes[meta["traj_idx"]]
            trajectory = {
                **meta,
                "episode_id": episode["episode_id"],
                "artifact": episode["artifact"],
            }
        rows.append(
            {
                "trajectory": trajectory,
                "datum": {
                    "model_input": sent.model_input.model_dump(mode="json"),
                    "loss_fn_inputs": {
                        name: tensor(value) for name, value in sent.loss_fn_inputs.items()
                    },
                    "model_input_spans": spans(sent.model_input_spans),
                    "loss_fn_input_spans": spans(sent.loss_fn_input_spans),
                },
                "cookbook_mask": tensor(raw.loss_fn_inputs["mask"])["data"]
                if "mask" in raw.loss_fn_inputs
                else None,
            }
        )
    content = (
        json.dumps(
            {"status": "prepared_for_dispatch", "loss_fn": "importance_sampling", "datums": rows},
            allow_nan=False,
        )
        + "\n"
    ).encode()
    compressed = gzip.compress(content, mtime=0)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".training-payload-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def implementation_record(config: TrainConfig, task, renderer=None) -> dict:
    """Record the public protocol and editable source identity, without credentials."""
    from elt_rl.environment import (
        initial_messages,
        instructions_for,
        uses_native_tools,
    )
    from elt_rl.history import (
        HISTORY_COMPACTION_RATIO,
        MODEL_HISTORY_POLICY_VERSION,
        REASONING_RESPONSE_RESERVE,
    )

    initial = {
        "instructions": instructions_for(task),
        **({"execution_backend": "native"} if uses_native_tools(task) else {}),
        "task": task.public(),
        "dialect": config.destination,
        "max_turns": config.max_turns,
    }
    if config.task_spec_path:
        from elt_rl.model_registry import MODELS

        messages = MODELS.get(config.model).codec.initial_messages(initial, renderer)
    else:
        messages = initial_messages(initial)
    initial["instructions"] = messages[0]["content"]
    encoded = json.dumps(initial, sort_keys=True, default=str).encode()
    encoded_messages = json.dumps(messages, sort_keys=True, default=str).encode()
    return {
        "action_protocol": config.action_protocol,
        "renderer_conditioning": {"renderer": config.renderer, "effort": config.renderer_effort},
        "model_history": {
            "type": "context_budget_compaction",
            "policy_version": MODEL_HISTORY_POLICY_VERSION,
            "response_headroom_tokens": min(
                REASONING_RESPONSE_RESERVE, (config.context_tokens - 1) // 4
            ),
            "compaction_target_fraction": list(HISTORY_COMPACTION_RATIO),
            "preserve_initial_observation": True,
            "preserve_visible_assistant_actions": True,
            "older_observations": "omit_only_when_context_exceeds_headroom_target",
            "stable_omissions_between_compactions": True,
            "raw_history": "complete",
        },
        "reward_version": configured_reward_version(config),
        "public_initial_observation": initial,
        "public_initial_messages": messages,
        "public_initial_messages_sha256": hashlib.sha256(encoded_messages).hexdigest(),
        "public_initial_observation_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_sha256": implementation_fingerprint(),
    }


def episode_summary_row(label, index, reward, raw, artifact):
    """Use the same report fields for new and inherited terminal episodes."""
    report = raw.get("report") or {}
    from elt_rl.metrics import episode_metrics

    return {
        **episode_metrics(raw),
        "episode_id": f"{label}-{index:03d}",
        "reward": reward,
        "training_reward": reward,
        "official_score": report.get("official_score"),
        "official_success": report.get("official_success"),
        "success": bool(report.get("success")),
        "task_id": raw.get("task"),
        **{key: report.get(key) for key in REWARD_METRICS},
        "pipeline_valid": report.get("pipeline_valid", False),
        "execution_success": report.get("execution_success", False),
        "training_eligible": report.get("training_eligible", True),
        "failure_type": report.get("failure_type"),
        "termination": raw.get("termination"),
        "runner_stop_reason": raw.get("runner_stop_reason"),
        "turns": len(raw.get("sampling", [])),
        "action_errors": sum("error" in t.get("observation", {}) for t in raw.get("trace", [])),
        "artifact": artifact,
    }


async def collect_group(
    config,
    renderer,
    sampler,
    budget,
    out,
    label,
    size,
    seed,
    *,
    cached_episodes=None,
    summary_provenance=None,
    policy_checkpoint=None,
):
    from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
    from tinker_cookbook.rl.rollouts import do_group_rollout

    from elt_rl.cookbook_env import ELTGroupBuilder
    from elt_rl.model_runtime import ModelPolicy, SeededGroupStrategy

    started = time.monotonic()
    cached, previous_seconds = {}, 0.0
    if cached_episodes is not None:
        if not label.startswith(("eval-batch-", "final-")):
            raise ValueError("Batch episode reuse is restricted to journaled evaluation")
        cached = cached_episodes
    pending_indices = [index for index in range(size) if index not in cached]
    task_spec_path = config.task_spec_path
    if config.task_groups:
        from elt_rl.task_scheduling import TaskSchedule

        task_spec_path = (
            TaskSchedule(config.task_groups, config.evaluation_task).select(label).specification
        )
    evaluating = label.startswith(("eval", "final"))
    limits = config.evaluation_limits if evaluating else {}

    def limit(name):
        return limits.get(name, getattr(config, name))

    builder = ELTGroupBuilder(
        task_path=config.task,
        renderer=renderer,
        destination=config.destination,
        group_size=len(pending_indices),
        max_turns=limit("max_turns"),
        max_seconds=limit("max_seconds"),
        # ModelPolicy caps each completion against its actual remaining context.
        # Reserving max_tokens here would reject useful history before sampling.
        context_tokens=limit("context_tokens") - 1,
        artifact_dir=str(out / "episodes"),
        artifact_prefix=label,
        max_episode_tokens=limit("max_episode_tokens"),
        sampling_timeout_seconds=limit("sampling_timeout_seconds"),
        allow_remote=config.allow_remote,
        episode_indices=tuple(pending_indices) if cached_episodes is not None else None,
        task_spec_path=task_spec_path,
        reward_config=config.reward,
        model_id=config.model,
        run_id=config.run_id,
        policy_checkpoint=policy_checkpoint,
    )

    def record_sampling(event):
        ledger_path = out / "run-ledger.json"
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {"fingerprint": fingerprint(config, load_configured_tasks(config))}
        )
        ledger.update(sampling=budget.public(), phase=label, time=time.time())
        write_json(ledger_path, ledger)
        with (out / "sampling-progress.jsonl").open("a") as file:
            file.write(json.dumps({"phase": label, "time": time.time(), **event}) + "\n")

    policy = ModelPolicy(
        sampler,
        budget,
        max_tokens=limit("max_tokens"),
        context_tokens=limit("context_tokens"),
        max_episode_tokens=limit("max_episode_tokens"),
        temperature=config.evaluation_temperature
        if evaluating and config.evaluation_temperature is not None
        else config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=seed,
        progress=record_sampling,
    )
    group = None
    new_rewards = []
    if pending_indices:
        group = await do_group_rollout(
            builder,
            policy,
            strategy=SeededGroupStrategy(concurrency=config.concurrency),
            termination=TerminationRewardPolicy(zero_reward_on_limit=True),
        )
        new_rewards = group.get_total_rewards()
    if len(new_rewards) != len(pending_indices) or not all(math.isfinite(r) for r in new_rewards):
        raise RuntimeError("Invalid or incomplete rollout reward group")
    rows_by_index = {
        index: episode_summary_row(
            label,
            index,
            evidence["raw"]["report"]["training_reward"],
            evidence["raw"],
            evidence["artifact"],
        )
        for index, evidence in cached.items()
    }
    for i, env in enumerate(builder._envs):
        index = getattr(env, "episode_index", i)
        if index not in pending_indices or index in rows_by_index:
            raise RuntimeError(
                "Evaluation episode indices overlap or differ from the requested seed set"
            )
        rows_by_index[index] = episode_summary_row(
            label,
            index,
            new_rewards[i],
            {
                "report": env.core.report,
                "termination": env.core.termination,
                "sampling": env.sampling_trace,
                "trace": env.core.trace,
                "runner_stop_reason": env.runner_stop_reason,
                "task": env.core.task.name,
            },
            evidence_ref(out / "episodes" / f"{label}-{index:03d}.json"),
        )
    if sorted(rows_by_index) != list(range(size)):
        raise RuntimeError(
            "Evaluation summary must contain every original episode index exactly once"
        )
    rows = [rows_by_index[index] for index in range(size)]
    from elt_rl.metrics import episode_metrics, group_metrics

    for row in rows:
        raw = json.loads(Path(row["artifact"]["artifact_path"]).read_text())
        row.update(episode_metrics(raw))
        if config.reward["type"] == "pure_cell" and row.get("cell_accuracy") is None:
            if raw.get("report") or not raw.get("done") or row["reward"] != 0:
                raise RuntimeError("Completed pure-cell rollout lacks its scoring receipt")
            if not row["training_eligible"] or str(raw.get("termination", "")).startswith(
                "infrastructure"
            ):
                raise RuntimeError("Infrastructure outcome cannot become a zero cell score")
            from elt_rl.metrics import zero_cell_results
            from elt_rl.task_spec import load_task_spec

            mask = zero_cell_results(load_task_spec(task_spec_path))
            row.update(
                cell_accuracy=0.0, correct_cells=0, required_cells=len(mask), cell_results=mask
            )
    rewards = [row["reward"] for row in rows]
    elapsed = time.monotonic() - started
    summary = {
        **group_metrics(rows),
        "label": label,
        "renderer_conditioning": {"renderer": config.renderer, "effort": config.renderer_effort},
        "seed": seed,
        "episodes": rows,
        "rewards": rewards,
        "successes": sum(row["success"] for row in rows),
        "n": size,
        "success_rate": sum(row["success"] for row in rows) / size,
        "mean_reward": sum(rewards) / size,
        "mean_official_score": (
            sum(row["official_score"] for row in rows) / size
            if all(row["official_score"] is not None for row in rows)
            else None
        ),
        "reward_version": configured_reward_version(config),
        "task_id": rows[0].get("task_id") if rows else None,
        "policy_checkpoint": policy_checkpoint,
        "reward_configuration": config.reward,
        "episode_seed_set": [(seed + i * 1009) % (2**31) for i in range(size)],
        "official_success_rate": sum(bool(row["official_success"]) for row in rows) / size,
        "seconds": previous_seconds + elapsed,
        "usage": budget.public(),
        **(summary_provenance or {}),
    }
    for key in ("column_score", "cell_score", "value_score"):
        values = [row.get(key) for row in rows]
        if all(value is not None for value in values):
            summary[key + "_values"] = values
            summary["mean_" + key] = sum(values) / len(values)
    if cached_episodes is not None:
        summary.update(reused_episode_indices=sorted(cached), new_episode_indices=pending_indices)
    if config.schedule == "batches_v1" and label.startswith("train-batch-"):
        from elt_rl.trajectory_archive import save_group

        summary["group_archive"] = save_group(out / f"{label}-group.json.gz", group)
    write_json(out / f"{label}.json", summary)
    return group, summary


def validate_metrics(metrics):
    for name, value in (metrics or {}).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite service metric: {name}")


async def apply_training_update(client, data, config, control, event, journal):
    """Record the distinct gradient computation and Adam service acknowledgments."""
    import tinker

    from elt_rl.errors import OutcomeUnknown

    control.check_dispatch()
    started = time.perf_counter()
    event["status"] = "forward_backward_dispatching"
    write_json(journal, event)
    try:
        async with asyncio.timeout(config.optimization_timeout_seconds):
            future = await client.forward_backward_async(data, loss_fn="importance_sampling")
            event["forward_backward_request_id"] = getattr(future, "request_id", None)
            write_json(journal, event)
            result = await future.result_async()
            event["status"] = "forward_backward_confirmed"
            # The SDK exposes a typed dataclass with one loss-output mapping per datum.
            event["forward_backward_result"] = {
                "loss_fn_output_type": result.loss_fn_output_type,
                "output_count": len(result.loss_fn_outputs),
                "metrics": result.metrics,
            }
            event["forward_backward_metrics"] = result.metrics
            validate_metrics(result.metrics)
            write_json(journal, event)
            control.check_dispatch()
            event["status"] = "optimizer_dispatching"
            write_json(journal, event)
            optim = await client.optim_step_async(
                tinker.AdamParams(
                    learning_rate=config.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                    weight_decay=0,
                    grad_clip_norm=1.0,
                )
            )
            event["optimizer_request_id"] = getattr(optim, "request_id", None)
            write_json(journal, event)
            receipt = await optim.result_async()
            event["status"] = "optimizer_service_confirmed"
            event["optimizer_metrics"] = receipt.metrics
            event["optimizer_result"] = receipt.model_dump(mode="json")
            event["optimization_seconds"] = time.perf_counter() - started
            write_json(journal, event)
            return receipt
    except BaseException as exc:
        phase = event["status"]
        event.update(failure_phase=phase, exception=type(exc).__name__)
        if phase == "forward_backward_confirmed":
            event["status"] = "optimizer_not_dispatched"
        elif phase != "optimizer_service_confirmed":
            event["status"] = "outcome_unknown"
        write_json(journal, event)
        if event["status"] == "outcome_unknown":
            raise OutcomeUnknown("Reconcile the service outcome before retrying") from exc
        raise


async def save_checkpoint(
    client, out, config, inputs, updates, attempts, *, lineage, ledger, resume_allowed=True
):
    from elt_rl.errors import OutcomeUnknown

    name = f"update-{updates:04d}-attempt-{attempts:04d}-{uuid.uuid4().hex[:8]}"
    journal = out / f"checkpoint-dispatch-{name}.json"
    event = {"status": "dispatching", "updates": updates, "attempts": attempts, "name": name}
    write_json(journal, event)
    try:
        async with asyncio.timeout(config.checkpoint_timeout_seconds):
            training = await client.save_state_async(name)
            event["training_state_request_id"] = getattr(training, "request_id", None)
            write_json(journal, event)
            state = await training.result_async()
            event.update(
                status="training_state_confirmed", state_result=state.model_dump(mode="json")
            )
            write_json(journal, event)
            sampling = await client.save_weights_for_sampler_async(name)
            event["sampler_state_request_id"] = getattr(sampling, "request_id", None)
            write_json(journal, event)
            sampler = await sampling.result_async()
        event.update(status="confirmed", sampler_result=sampler.model_dump(mode="json"))
        write_json(journal, event)
    except BaseException as exc:
        detail = str(exc)
        if key := os.environ.get("TINKER_API_KEY"):
            detail = detail.replace(key, "[REDACTED]")
        event.update(status="outcome_unknown", exception=type(exc).__name__, error=detail[:3000])
        write_json(journal, event)
        raise OutcomeUnknown(
            "Checkpoint save outcome requires reconciliation; no automatic retry. " + detail[:3000]
        ) from exc
    saved = {
        "path": state.path,
        "sampler_path": sampler.path,
        "updates": updates,
        "attempts_completed": attempts,
        "config": asdict(config),
        "fingerprint": inputs,
        "lineage": lineage,
        "budget_ledger": ledger,
        "save_receipt": evidence_ref(journal),
        "resume_allowed": resume_allowed,
    }
    filename = (
        f"checkpoint-{updates:04d}.json"
        if resume_allowed
        else f"reconciliation-state-{attempts:04d}.json"
    )
    write_json(out / filename, saved)
    os.chmod(out / filename, 0o600)
    if resume_allowed:
        write_json(out / "checkpoint.json", saved)
        os.chmod(out / "checkpoint.json", 0o600)
    return saved


async def train(config: TrainConfig):
    from elt_rl.batch_protocol import train_batches

    return await train_batches(config)


async def evaluate_model(config: TrainConfig, checkpoint: str | None = None):
    task = preflight(config, training=False)
    import tinker

    from elt_rl.model_runtime import TokenBudget, resolve_renderer

    renderer_name, renderer = resolve_renderer(
        config.model, config.renderer, config.renderer_effort
    )
    config = replace(config, renderer=renderer_name)
    inputs = fingerprint(config, task)
    saved = checked_checkpoint(checkpoint, config, inputs, for_evaluation=True)
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "run-config.json",
        {
            "config": asdict(config),
            "versions": versions(),
            "fingerprint": inputs,
            "checkpoint": checkpoint,
            "purpose": "evaluation",
            "training_destination": saved["config"]["destination"] if saved else None,
            "cross_destination_evaluation": bool(
                saved and saved["config"]["destination"] != config.destination
            ),
            "implementation": implementation_record(config, task, renderer),
        },
    )
    from elt_rl.control import deadline_epoch

    budget = TokenBudget(
        config.max_sampling_tokens,
        deadline_epoch=deadline_epoch(config.deadline_utc),
    )
    from elt_rl.tinker_client import harden_sampler, harden_service, no_retry_config

    service = tinker.ServiceClient(display_name="elt-bench-rl-eval")
    status = "errored"
    try:
        await harden_service(service, read_timeout_seconds=config.sampling_timeout_seconds)
        sampler = await service.create_sampling_client_async(
            **({"model_path": saved["sampler_path"]} if saved else {"base_model": config.model}),
            retry_config=no_retry_config(),
        )
        write_json(
            out / "sampler-dispatch-policy.json",
            await harden_sampler(sampler, read_timeout_seconds=config.sampling_timeout_seconds),
        )
        _, summary = await collect_group(
            config,
            renderer,
            sampler,
            budget,
            out,
            "evaluation",
            config.eval_episodes,
            (config.seed + 100_000) % (2**31),
        )
        if budget.exhausted:
            raise RuntimeError("Sampling token budget exhausted; evaluation incomplete")
        if fingerprint(config, task) != inputs:
            raise RuntimeError("Task/source/reference files changed during evaluation")
        status = "success"
        return summary
    except TokenLimitExceeded:
        status = "stopped_budget"
        raise
    finally:
        try:
            write_json(
                out / "run-status.json",
                {"status": status, "optimizer_steps": 0, "sampling": budget.public()},
            )
        finally:
            await service.close(
                "success"
                if status == "success"
                else "interrupted"
                if status == "stopped_budget"
                else "errored"
            )
