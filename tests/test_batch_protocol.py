"""Offline protocol proofs using actual Cookbook tensors, not experiment evidence."""

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

from elt_rl import batch_protocol
from elt_rl import train as recipe
from elt_rl.artifacts import write_json
from elt_rl.lineage import evidence_ref
from elt_rl.trajectory_archive import save_group


@pytest.fixture
def batch_case(tmp_path, monkeypatch):
    from protocol_test_support import Result, future

    calls, zeros = [], set()
    flags = {
        "unknown_optimizer": False,
        "interrupt_summary": None,
        "interrupt_save": None,
        "expensive_baseline": False,
    }
    task = SimpleNamespace(
        fingerprint=lambda: {"fixture": "offline-batch-protocol"},
        public=lambda: {"name": "offline-only"},
    )
    monkeypatch.setattr(recipe, "preflight", lambda config: task)
    monkeypatch.setattr("elt_rl.model_runtime.resolve_renderer", lambda *a: ("tml_v0", object()))
    monkeypatch.setattr("elt_rl.reporting.render_run_report", lambda *a: {})

    def forbidden_administration(*args, **kwargs):
        raise AssertionError("Training must not query quotas or require checkpoint inventory")

    monkeypatch.setattr(tinker, "AdminClient", forbidden_administration)
    monkeypatch.setattr(tinker, "RestClient", forbidden_administration)
    monkeypatch.setattr(
        "elt_rl.tinker_client.harden_service", AsyncMock(return_value={"offline": True})
    )
    monkeypatch.setattr("elt_rl.tinker_client.harden_sampler", AsyncMock(return_value={}))

    class Client:
        version = 0

        async def forward_backward_async(self, data, loss_fn):
            calls.append(("gradient", self.version, len(data)))
            assert loss_fn == "importance_sampling"
            for datum in data:
                assert set(datum.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
            return future(
                tinker.types.ForwardBackwardOutput(
                    loss_fn_output_type=loss_fn,
                    loss_fn_outputs=[{} for _ in data],
                    metrics={"loss:sum": 0.1},
                )
            )

        async def optim_step_async(self, params):
            self.version += 1
            calls.append(("optimizer", self.version))
            assert params.learning_rate == 1e-5
            if flags["unknown_optimizer"]:
                raise TimeoutError("Offline injected lost optimizer response")
            return future(Result(metrics={"unclipped_grad_l2:mean": 1.0}))

        async def save_state_async(self, name):
            return future(Result(path="fixture://state/" + str(self.version)))

        async def save_weights_for_sampler_async(self, name):
            return future(Result(path="fixture://sampler/" + str(self.version)))

    client = Client()

    class Service:
        def __init__(self, **kwargs):
            calls.append(("service",))
            self.holder = SimpleNamespace(_session_id="offline-unit-test")

        async def get_server_capabilities_async(self):
            return SimpleNamespace(
                supported_models=[
                    tinker.types.SupportedModel(
                        model_name=config.model, trainable=True, sampleable=True
                    )
                ]
            )

        async def create_lora_training_client_async(self, **kwargs):
            client.version = 0
            return client

        async def create_training_client_from_state_with_optimizer_async(self, path):
            client.version = int(path.rsplit("/", 1)[1])
            return client

        async def create_sampling_client_async(self, model_path, **kwargs):
            return SimpleNamespace(version=int(model_path.rsplit("/", 1)[1]), model_path=model_path)

        async def close(self, status):
            calls.append(("close", status))

    monkeypatch.setattr(tinker, "ServiceClient", Service)

    async def collect(config, renderer, sampler, budget, out, label, size, seed):
        calls.append(("collect", label, size, sampler.version, seed))
        train = label.startswith("train-batch")
        batch = int(label.rsplit("-", 1)[1]) if train else None
        rewards = [
            (0.2 if batch in zeros or i % 2 == 0 else 0.8) if train else 0.4 for i in range(size)
        ]
        trajectories, rows = [], []
        for i, reward in enumerate(rewards):
            from elt_rl.rewards import (
                RewardConfig,
                RewardContext,
                RewardField,
                RewardTable,
                RewardTarget,
                score_execution,
            )

            # Synthetic protocol trajectories still carry a genuine scorer receipt.
            fields = tuple(
                RewardField(f"c{j}", "integer", "identifier" if j == 0 else "business")
                for j in range(5)
            )
            target = RewardTarget("offline_fixture", fields, ("c0",))
            reference = RewardTable(tuple(f.name for f in fields), ((1, 1, 1, 1, 1),))
            prediction = replace(
                reference, rows=(tuple(1 if j < round(reward * 5) else 0 for j in range(5)),)
            )
            scored = score_execution(
                RewardContext(
                    "offline",
                    (target,),
                    {target.name: prediction},
                    {target.name: reference},
                    {},
                    True,
                    True,
                ),
                RewardConfig.from_dict(config.reward),
            )
            selection_sha = hashlib.sha256(
                json.dumps(config.reward, sort_keys=True).encode()
            ).hexdigest()
            scored.update(selection_sha256=selection_sha, targets_snapshot={})
            score_path = out / "synthetic-scores" / f"{label}-{i:03d}.json"
            write_json(score_path, scored)
            report = {
                "training_reward": reward,
                "reward_type": scored["reward_type"],
                "reward_version": scored["version"],
                "cell_accuracy": scored["cell_accuracy"],
                "cell_results": scored["cell_results"],
                "reward_result": {
                    "artifact": evidence_ref(score_path),
                    "config_sha256": scored["config_sha256"],
                    "selection_sha256": selection_sha,
                },
            }
            count = 800000 if flags["expensive_baseline"] and label == "eval-batch-0000" else 2
            assert budget.reserve(count + 2)
            budget.settle(count + 2, count, 2)
            raw = {
                "done": True,
                "termination": "submitted",
                "report": report,
                "fingerprint": task.fingerprint(),
                "implementation_sha256": recipe.fingerprint(config, task)["implementation_sha256"],
                "episode_id": f"{label}-{i:03d}",
                "sampling": [
                    {
                        "status": "sampled",
                        "seed": (seed + 1009 * i) % 2**31,
                        "temperature": config.temperature,
                        "tokens": [10 + i, 99],
                        "logprobs": [-0.5, -0.5],
                    }
                ],
            }
            path = out / "episodes" / f"{label}-{i:03d}.json"
            write_json(path, raw)
            rows.append(
                {
                    "episode_id": raw["episode_id"],
                    "artifact": evidence_ref(path),
                    "reward": reward,
                    "training_eligible": True,
                    "success": False,
                }
            )
            trajectories.append(
                Trajectory(
                    [
                        Transition(
                            tinker.ModelInput.from_ints([1, 2]),
                            TokensWithLogprobs(
                                tokens=[10 + i, 99],
                                maybe_logprobs=[-0.5, -0.5],
                                sequence_id=f"offline-{label}-{i}",
                            ),
                            reward,
                            True,
                        )
                    ],
                    tinker.ModelInput.from_ints([]),
                )
            )
        group = TrajectoryGroup(trajectories, [0.0] * size, [{} for _ in range(size)])
        summary = {
            "label": label,
            "n": size,
            "seed": seed,
            "episode_seed_set": [(seed + 1009 * i) % 2**31 for i in range(size)],
            "episodes": rows,
            "rewards": rewards,
            "mean_reward": sum(rewards) / size,
            "successes": 0,
            "matched_columns": [0] * size,
            "success_rate": 0,
            "seconds": 0,
        }
        if train:
            summary["group_archive"] = save_group(out / f"{label}-group.json.gz", group)
        write_json(out / f"{label}.json", summary)
        if label == flags["interrupt_summary"]:
            flags["interrupt_summary"] = None
            raise asyncio.CancelledError
        return group, summary

    monkeypatch.setattr(recipe, "collect_group", collect)
    original_save = recipe.save_checkpoint

    async def save(*args, **kwargs):
        result = await original_save(*args, **kwargs)
        if result["updates"] == flags["interrupt_save"]:
            flags["interrupt_save"] = None
            raise asyncio.CancelledError
        return result

    monkeypatch.setattr(recipe, "save_checkpoint", save)
    config = recipe.TrainConfig(
        "offline",
        str(tmp_path / "run"),
        schedule="batches_v1",
        batches=16,
        group_size=8,
        eval_episodes=10,
        final_eval_episodes=20,
        temperature=0.8,
        concurrency=8,
    )
    return config, calls, zeros, flags


async def test_approved_schedule_218_and_zero_signal_batches(batch_case):
    config, calls, zeros, _ = batch_case
    zeros.update({4, 12})
    result = await recipe.train(config)
    collects = [call for call in calls if call[0] == "collect"]
    assert sum(call[2] for call in collects) == 218
    assert [call[1] for call in collects if call[1].startswith("eval-")] == [
        f"eval-batch-{i:04d}" for i in [0, 4, 8, 12, 16]
    ]
    assert result["attempted_batches"] == result["completed_batches"] == 16
    assert result["updates"] == 14 == sum(call[0] == "optimizer" for call in calls)
    assert collects[-2][3] == 0 and collects[-1][3] == 14
    assert collects[-2][4] == collects[-1][4] == config.final_seed
    assert config.final_seed != collects[0][4]
    # Evaluation never supplies a gradient; precisely the 14 nonconstant training groups do.
    assert sum(call[0] == "gradient" for call in calls) == 14
    before = calls.copy()
    resumed = await recipe.train(
        replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0014.json"))
    )
    assert resumed["status"] == "complete" and calls == before


async def test_confirmed_checkpoint_crash_resumes_without_duplicate_update(batch_case):
    config, calls, _, flags = batch_case
    flags["interrupt_save"] = 1
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(config)
    result = await recipe.train(
        replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0000.json"))
    )
    assert result["updates"] == 16
    assert sum(call[0] == "optimizer" for call in calls) == 16
    assert sum(call[2] for call in calls if call[0] == "collect") == 218


async def test_confirmed_base_checkpoint_crash_recovers_without_extra_samples(batch_case):
    config, calls, _, flags = batch_case
    flags["interrupt_save"] = 0
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(config)
    result = await recipe.train(
        replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0000.json"))
    )
    assert result["updates"] == 16
    assert sum(call[0] == "optimizer" for call in calls) == 16
    assert sum(call[2] for call in calls if call[0] == "collect") == 218


async def test_update_only_crash_recovery_keeps_evaluation_disabled(batch_case):
    config, calls, _, flags = batch_case
    config = replace(config, batches=1, evaluation_interval=1, evaluation_enabled=False)
    flags["interrupt_save"] = 1
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(config)
    result = await recipe.train(
        replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0000.json"))
    )
    assert result["status"] == "complete" and result["updates"] == 1
    assert result["completed_batches"] == 1 and not result["final_evaluations"]
    assert not result["process_evaluations"]
    assert [call[1] for call in calls if call[0] == "collect"] == ["train-batch-0001"]
    assert sum(call[0] == "optimizer" for call in calls) == 1


def test_newer_settled_sampling_snapshot_does_not_retain_old_inflight_hold():
    previous = {"input_tokens": 10, "output_tokens": 0, "reserved_or_uncertain_tokens": 100}
    observed = {"input_tokens": 12, "output_tokens": 2, "reserved_or_uncertain_tokens": 0}
    state = {"sampling": previous, "recorded_at": 10}
    latest = {"fingerprint": {"unit": True}, "time": 11, "sampling": observed}
    assert batch_protocol.recover_sampling(state, latest, {"unit": True}) == observed


async def test_final_report_interruption_resumes_without_remote_calls(batch_case, monkeypatch):
    config, calls, _, _ = batch_case
    interrupted = False

    def report(out, state):
        nonlocal interrupted
        if state.get("phase") == "complete" and not interrupted:
            interrupted = True
            raise asyncio.CancelledError
        return {}

    monkeypatch.setattr("elt_rl.reporting.render_run_report", report)
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(config)
    before = calls.copy()
    result = await recipe.train(
        replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0016.json"))
    )
    assert result["status"] == "complete" and calls == before
    assert set(result["final_evaluations"]) == {"base", "final"}


async def test_completed_eval_recovery_preserves_samples_and_batch_numbers(batch_case):
    config, calls, _, flags = batch_case
    flags["interrupt_summary"] = "eval-batch-0004"
    with pytest.raises(asyncio.CancelledError):
        await recipe.train(config)
    result = await recipe.train(
        replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0004.json"))
    )
    assert result["completed_batches"] == result["attempted_batches"] == result["updates"] == 16
    assert sum(call[2] for call in calls if call[0] == "collect") == 218


async def test_unknown_optimizer_outcome_never_retried(batch_case):
    from elt_rl.errors import OutcomeUnknown

    config, calls, _, flags = batch_case
    flags["unknown_optimizer"] = True
    with pytest.raises(OutcomeUnknown):
        await recipe.train(config)
    with pytest.raises(ValueError, match="reconciliation"):
        await recipe.train(
            replace(config, resume_checkpoint=str(Path(config.output) / "checkpoint-0000.json"))
        )
    assert sum(call[0] == "optimizer" for call in calls) == 1
