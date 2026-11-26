"""Approved batch scheduling around the existing Cookbook/Tinker execution functions.

No alternative learning algorithm: collection, advantage computation, tensor masks,
forward/backward, Adam, checkpoint saves and SDK hardening reuse the existing recipe.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.control import RunControl
from elt_rl.errors import InfrastructureError, IntegrityError, OutcomeUnknown, TokenLimitExceeded
from elt_rl.lineage import evidence_ref, positive_gradient, read_evidence, verify_lineage


def validate_cohort(summary, *, label, size, seed, inputs):
    from elt_rl.task_scheduling import expected_task_fingerprint

    if (
        summary.get("label") != label
        or summary.get("n") != size
        or summary.get("seed") != seed
        or len(summary.get("episodes", [])) != size
        or len(summary.get("rewards", [])) != size
        or summary.get("episode_seed_set") != [(seed + 1009 * i) % 2**31 for i in range(size)]
    ):
        raise IntegrityError("Cohort does not match its immutable sample/seed plan")
    for i, row in enumerate(summary["episodes"]):
        if row.get("episode_id") != f"{label}-{i:03d}" or not row.get("training_eligible", True):
            raise IntegrityError("Incomplete or ineligible trajectory cannot enter a cohort")
        raw = read_evidence(row["artifact"])
        if not raw.get("done") or str(raw.get("termination", "")).startswith("infrastructure"):
            raise IntegrityError("Infrastructure failure is not a policy zero-reward trajectory")
        if row["reward"] != summary["rewards"][i]:
            raise IntegrityError("Cohort reward identity mismatch")
        if raw.get("episode_id") != row["episode_id"] or raw.get(
            "fingerprint"
        ) != expected_task_fingerprint(inputs, label):
            raise IntegrityError("Cohort contains a trajectory from different frozen inputs")
        if raw.get("implementation_sha256") != inputs["implementation_sha256"]:
            raise IntegrityError("Cohort trajectory source differs from the frozen implementation")
        report = raw.get("report") or {}
        if inputs.get("reward_configuration") is not None and raw.get("termination") == "submitted":
            from elt_rl.execution_scoring import verify_scoring_receipt

            verify_scoring_receipt(raw, inputs["reward_configuration"])
        if (
            not report.get("training_eligible", True)
            or report.get("failure_type") == "infrastructure"
        ):
            raise IntegrityError("Infrastructure grading cannot enter training/evaluation metrics")
        if report.get("training_reward", 0) != row["reward"]:
            raise IntegrityError("Trajectory reward differs from its execution report")
        for turn, event in enumerate(raw.get("sampling", [])):
            if event.get("stop_reason") in {"run_token_budget", "episode_token_budget"}:
                if event["stop_reason"] == "run_token_budget":
                    raise TokenLimitExceeded("Global sampling token limit reached")
                continue
            if (
                event.get("status") != "sampled"
                or event.get("seed") != (seed + 1009 * i + turn) % 2**31
            ):
                raise IntegrityError("Cohort sampling receipt/seed differs from its episode plan")
            sampling_protocol = (
                inputs.get("sampling_protocol", inputs["evaluation_protocol"])
                if label.startswith("train")
                else inputs["evaluation_protocol"]
            )
            if event.get("temperature") != sampling_protocol["temperature"]:
                raise IntegrityError("Sampling temperature differs from frozen protocol")


def validate_training_group(group, summary):
    """Bind the exact Cookbook actions/logprobs to their executed episode artifacts."""
    if len(group.trajectories_G) != len(summary["episodes"]):
        raise IntegrityError("Training group cardinality differs from executed episodes")
    for trajectory, row in zip(group.trajectories_G, summary["episodes"], strict=True):
        raw = read_evidence(row["artifact"])
        sampled = [
            event
            for event in raw.get("sampling", [])
            if event.get("status") == "sampled" and event.get("tokens")
        ]
        actions = [transition.ac for transition in trajectory.transitions if transition.ac.tokens]
        if len(actions) != len(sampled) or any(
            action.tokens != event.get("tokens") or action.logprobs != event.get("logprobs")
            for action, event in zip(actions, sampled, strict=True)
        ):
            raise IntegrityError(
                "Cookbook action tokens/logprobs differ from executed model output"
            )


async def train_batches(config):
    from elt_rl import train as recipe
    from elt_rl.model_runtime import TokenBudget, resolve_renderer

    task = recipe.preflight(config)
    renderer_name, renderer = resolve_renderer(
        config.model, config.renderer, config.renderer_effort
    )
    config = replace(config, renderer=renderer_name)
    inputs = recipe.fingerprint(config, task)
    out = Path(config.output).resolve()
    if not config.resume_checkpoint:
        out.mkdir(parents=True, exist_ok=False, mode=0o700)
    elif not out.is_dir():
        raise ValueError("Batch recovery must use the original run directory")
    with (out / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InfrastructureError("Another controller owns this experiment") from exc
        return await _run(config, task, renderer, inputs, out, TokenBudget)


def reconcile_local_checkpoint(state, out, inputs):
    """Recover a confirmed save after controller death, without another remote update."""
    batch = state["completed_batches"] + 1
    journal = out / f"optimization-batch-{batch:04d}.json"
    if not journal.exists():
        return None
    event = json.loads(journal.read_text())
    updates = event.get("updates")
    if (
        event.get("status")
        not in {"effective_update_confirmed", "effective_update_confirmed_checkpoint_pending"}
        or updates != state["updates"] + 1
    ):
        return None
    path = out / f"checkpoint-{updates:04d}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    verify_lineage(saved)
    if (
        saved["fingerprint"] != inputs
        or saved["lineage"]["parent_checkpoint"] != state["current_checkpoint"]
    ):
        raise IntegrityError("Recovered checkpoint does not extend the current policy")
    receipt = read_evidence(saved["lineage"]["optimizer_receipt"])
    if receipt.get("batch") != batch or receipt.get("optimizer_request_id") != event.get(
        "optimizer_request_id"
    ):
        raise IntegrityError("Recovered checkpoint refers to a different optimizer dispatch")
    reference = evidence_ref(path)
    event.update(status="effective_update_confirmed", checkpoint=reference)
    write_json(journal, event)
    state.update(updates=updates, current_checkpoint=reference)
    state.setdefault("recovery_events", []).append(
        {
            "time": time.time(),
            "kind": "confirmed_checkpoint_local_finalize",
            "batch": batch,
            "checkpoint": reference,
        }
    )
    write_json(out / "protocol-state.json", state)
    return str(path)


def recover_sampling(state, latest, inputs):
    previous = state.get("sampling", {})
    if latest is None:
        return previous
    if latest.get("fingerprint") != inputs:
        raise IntegrityError("Sampling journal belongs to different frozen inputs")
    if latest.get("time", 0) <= state.get("recorded_at", 0):
        return previous
    observed = latest.get("sampling", {})
    if any(observed.get(k, 0) < previous.get(k, 0) for k in ("input_tokens", "output_tokens")):
        raise IntegrityError("Confirmed sampling counters decreased in the newer journal")
    # A settled request releases unused maximum-output capacity. Mixing maxima
    # from different snapshots would double-count that already-settled hold.
    return observed


async def _run(config, task, renderer, inputs, out, token_budget_type):
    import tinker
    from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

    from elt_rl import train as recipe
    from elt_rl.reporting import render_run_report
    from elt_rl.tinker_client import harden_sampler, harden_service, no_retry_config
    from elt_rl.trajectory_archive import load_group

    state_file = out / "protocol-state.json"
    stopped_status = (
        "token_limit_stopped_final_evaluated"
        if config.evaluation_enabled
        else "token_limit_stopped"
    )
    resumed = config.resume_checkpoint is not None
    if resumed:
        state = json.loads(state_file.read_text())
        if state.get("fingerprint") != inputs or state.get("format") != "elt_batch_protocol_v1":
            raise IntegrityError("Recovery code, inputs, budget or approved protocol changed")
        recovered_checkpoint = reconcile_local_checkpoint(state, out, inputs)
        if recovered_checkpoint:
            config = replace(config, resume_checkpoint=recovered_checkpoint)
        current = recipe.checked_checkpoint(config.resume_checkpoint, config, inputs)
        if "current_checkpoint" not in state:
            if current["updates"] != 0 or state["attempted_batches"] != 0:
                raise IntegrityError(
                    "Uninitialized state can recover only its confirmed base checkpoint"
                )
            state["current_checkpoint"] = evidence_ref(config.resume_checkpoint)
            state["base_checkpoint"] = state["current_checkpoint"]
        if evidence_ref(config.resume_checkpoint) != state["current_checkpoint"]:
            raise IntegrityError(
                "Resume the latest confirmed checkpoint referenced by protocol-state"
            )
        verify_lineage(current)
        if current["updates"] != state["updates"]:
            raise IntegrityError("Optimizer/checkpoint counters require reconciliation")
        expected_final = {"base", "final"} if config.evaluation_enabled else set()
        if state.get("phase") == "complete" or (
            config.evaluation_enabled and set(state.get("final_evaluations", {})) == expected_final
        ):
            selection = state.get("final_selection", {})
            if (
                set(state.get("final_evaluations", {})) != expected_final
                or selection.get("base") != state["base_checkpoint"]
                or selection.get("final") != state["current_checkpoint"]
            ):
                raise IntegrityError("Completed run is missing consistent final policy evidence")
            for name, summary in state["final_evaluations"].items():
                if summary.get("checkpoint") != selection[name]:
                    raise IntegrityError("Final outcomes belong to a different selected model")
                validate_cohort(
                    summary,
                    label=f"final-{name}",
                    size=config.final_eval_episodes,
                    seed=config.final_seed,
                    inputs=inputs,
                )
            # Final outcomes are durable; finish local reporting without remote requests.
            state.update(
                phase="complete",
                status="complete"
                if state["completed_batches"] == config.batches
                else stopped_status,
                recorded_at=time.time(),
            )
            write_json(state_file, state)
            write_json(
                out / "run-status.json",
                {
                    key: state.get(key)
                    for key in (
                        "status",
                        "phase",
                        "attempted_batches",
                        "completed_batches",
                        "updates",
                        "stop_reason",
                        "sampling",
                    )
                },
            )
            render_run_report(out, state)
            return state
        state["status"] = "running"
    else:
        state = {
            "format": "elt_batch_protocol_v1",
            "fingerprint": inputs,
            "config": asdict(config),
            "status": "running",
            "phase": "initialization",
            "attempted_batches": 0,
            "completed_batches": 0,
            "updates": 0,
            "batches": [],
            "process_evaluations": [],
            "final_evaluations": {},
            "cohorts": {},
            "started_at": time.time(),
            "stop_reason": None,
        }
        write_json(
            out / "run-config.json",
            {
                "config": asdict(config),
                "fingerprint": inputs,
                "versions": recipe.versions(),
                "implementation": recipe.implementation_record(config, task, renderer),
            },
        )
    latest_path = out / "run-ledger.json"
    previous = recover_sampling(
        state,
        json.loads(latest_path.read_text()) if resumed and latest_path.exists() else None,
        inputs,
    )
    control = RunControl(steps=config.batches, deadline_utc=config.deadline_utc)
    budget = token_budget_type(
        config.max_sampling_tokens,
        input_tokens=previous.get("input_tokens", 0),
        output_tokens=previous.get("output_tokens", 0),
        reserved=previous.get("reserved_or_uncertain_tokens", 0),
        deadline_epoch=control.deadline,
    )

    def persist():
        state.update(sampling=budget.public(), recorded_at=time.time())
        write_json(state_file, state)
        write_json(
            out / "run-status.json",
            {
                key: state.get(key)
                for key in (
                    "status",
                    "phase",
                    "attempted_batches",
                    "completed_batches",
                    "updates",
                    "stop_reason",
                    "sampling",
                )
            },
        )

    persist()
    service = tinker.ServiceClient(display_name=config.run_id or out.name)
    status = "errored"
    try:
        write_json(
            out / "sdk-dispatch-policy.json",
            await harden_service(service, read_timeout_seconds=config.sampling_timeout_seconds),
        )
        session_id = service.holder._session_id
        if not session_id:
            raise InfrastructureError("Cannot attribute Tinker charges to this experiment")
        if session_id not in state.setdefault("service_sessions", []):
            state["service_sessions"].append(session_id)
        persist()
        capabilities = await service.get_server_capabilities_async()
        models = [row for row in capabilities.supported_models if row.model_name == config.model]
        if len(models) != 1 or not models[0].trainable or not models[0].sampleable:
            raise InfrastructureError("Exact approved model is not trainable and sampleable")
        write_json(out / "model-capability.json", models[0].model_dump(mode="json"))
        client = (
            await service.create_training_client_from_state_with_optimizer_async(current["path"])
            if resumed
            else await service.create_lora_training_client_async(
                base_model=config.model, rank=config.rank, seed=config.seed
            )
        )

        async def sampler_for(saved):
            sampler = await service.create_sampling_client_async(
                model_path=saved["sampler_path"], retry_config=no_retry_config()
            )
            await harden_sampler(sampler, read_timeout_seconds=config.sampling_timeout_seconds)
            return sampler

        async def checkpoint(updates, event=None, receipt=None):
            curve = out / f"process-through-update-{updates:04d}.json"
            write_json(curve, state["process_evaluations"])
            return await recipe.save_checkpoint(
                client,
                out,
                config,
                inputs,
                updates,
                state["attempted_batches"],
                lineage={
                    "schedule": "batches_v1",
                    "parent_checkpoint": state.get("current_checkpoint") if updates else None,
                    "optimizer_receipt": receipt,
                    "curve_snapshot": evidence_ref(curve),
                    "evaluation_pending": False,
                },
                ledger={
                    "sampling": budget.public(),
                    "attempts": state["attempted_batches"],
                    "completed_batches": state["completed_batches"],
                    "training_input_tokens": state.get("training_input_tokens", 0),
                },
            )

        if not resumed:
            current = await checkpoint(0)
            state["current_checkpoint"] = evidence_ref(out / "checkpoint-0000.json")
            state["base_checkpoint"] = state["current_checkpoint"]
            persist()
        sampler = await sampler_for(current)

        async def cohort(label, size, seed, sampler, policy_ref=None):
            cached, recovery_ref = None, None
            policy_ref = policy_ref or state["current_checkpoint"]
            known = state["cohorts"].get(label)
            if known and known.get("checkpoint") != policy_ref:
                raise IntegrityError("Cohort belongs to another sampling policy")
            if known and known.get("status") == "complete":
                summary = read_evidence(known["summary"])
                validate_cohort(summary, label=label, size=size, seed=seed, inputs=inputs)
                group = (
                    load_group(summary["group_archive"]) if summary.get("group_archive") else None
                )
                return group, summary, True
            if known and known.get("status") == "dispatching":
                # Never restart a partially executed cohort and exceed the rollout plan.
                # A complete summary was committed only after its training archive.
                path = out / f"{label}.json"
                if path.exists():
                    summary = json.loads(path.read_text())
                    validate_cohort(summary, label=label, size=size, seed=seed, inputs=inputs)
                    known.update(status="complete", summary=evidence_ref(path))
                    persist()
                    return (
                        load_group(summary["group_archive"])
                        if summary.get("group_archive")
                        else None,
                        summary,
                        True,
                    )
                from elt_rl.batch_recovery import prepare_partial

                cached, recovery_ref = prepare_partial(
                    out,
                    known,
                    label=label,
                    size=size,
                    seed=seed,
                    inputs=inputs,
                    config=config,
                )
            state["phase"] = label
            state["cohorts"][label] = {
                **(known or {}),
                "status": "dispatching",
                "size": size,
                "seed": seed,
                "sampler_path": getattr(sampler, "model_path", None),
                "checkpoint": policy_ref,
            }
            persist()
            try:
                group, summary = await recipe.collect_group(
                    config,
                    renderer,
                    sampler,
                    budget,
                    out,
                    label,
                    size,
                    seed,
                    **({"policy_checkpoint": policy_ref} if config.task_spec_path else {}),
                    **(
                        {
                            "cached_episodes": cached,
                            "summary_provenance": {
                                "partial_evaluation_recovery": recovery_ref,
                            },
                        }
                        if cached is not None
                        else {}
                    ),
                )
                if recovery_ref:
                    summary["partial_evaluation_recovery"] = recovery_ref
                write_json(out / f"{label}.json", summary)
                if budget.exhausted:
                    raise TokenLimitExceeded("Global sampling token limit reached")
                validate_cohort(summary, label=label, size=size, seed=seed, inputs=inputs)
            finally:
                persist()
            state["cohorts"][label].update(
                status="complete", summary=evidence_ref(out / f"{label}.json")
            )
            persist()
            return group, summary, False

        async def evaluate_batch(batch):
            if any(node["batch"] == batch for node in state["process_evaluations"]):
                return
            label = f"eval-batch-{batch:04d}"
            _, summary, _ = await cohort(
                label, config.eval_episodes, (config.seed + 100_000) % 2**31, sampler
            )
            state["process_evaluations"].append(
                {
                    **summary,
                    "batch": batch,
                    "step": batch,
                    "updates": state["updates"],
                    "checkpoint": state["current_checkpoint"],
                    "same_policy_as_previous": bool(
                        state["process_evaluations"]
                        and state["process_evaluations"][-1]["updates"] == state["updates"]
                    ),
                }
            )
            persist()
            render_run_report(out, state)

        # Pending evaluations run before a further training dispatch, including after resume.
        if (
            config.evaluation_enabled
            and state["completed_batches"] % config.evaluation_interval == 0
        ):
            try:
                await evaluate_batch(state["completed_batches"])
            except TokenLimitExceeded as exc:
                state["stop_reason"] = "Process evaluation token limit reached: " + str(exc)
                persist()
        while state["completed_batches"] < config.batches and not state["stop_reason"]:
            control.check_dispatch()
            batch = state["completed_batches"] + 1
            label = f"train-batch-{batch:04d}"
            journal = out / f"optimization-batch-{batch:04d}.json"
            # An optimizer and checkpoint may be durable before local batch finalization.
            # Finalize that event locally; never dispatch its gradient a second time.
            if journal.is_file():
                prior = json.loads(journal.read_text())
                terminal = prior.get("status") in {
                    "effective_update_confirmed",
                    "skipped_zero_advantage",
                }
                if terminal:
                    if prior.get("batch") != batch or (
                        prior["status"] == "effective_update_confirmed"
                        and (
                            prior.get("updates") != state["updates"]
                            or prior.get("checkpoint") != state["current_checkpoint"]
                        )
                    ):
                        raise IntegrityError(
                            "Completed optimization differs from the recovery checkpoint"
                        )
                    state["completed_batches"] = batch
                    state["attempted_batches"] = max(batch, state["attempted_batches"])
                    state["batches"].append(prior)
                    persist()
                    if config.evaluation_enabled and batch % config.evaluation_interval == 0:
                        await evaluate_batch(batch)
                    continue
            state["attempted_batches"] = max(state["attempted_batches"], batch)
            persist()
            event = {
                "batch": batch,
                "attempt": state["attempted_batches"],
                "updates_before": state["updates"],
            }
            try:
                group, summary, recovered = await cohort(
                    label,
                    config.group_size,
                    (config.seed + 1_000_000 + (batch - 1) * 100_000) % 2**31,
                    sampler,
                )
                rewards = group.get_total_rewards()
                validate_training_group(group, summary)
                if rewards != summary["rewards"]:
                    raise IntegrityError("Archived group rewards differ from executed cohort")
                advantages = compute_advantages([group])
                if max(rewards) == min(rewards):
                    advantages[0].zero_()
                event.update(
                    rewards=rewards,
                    advantages=advantages[0].tolist(),
                    mean_reward=summary["mean_reward"],
                    rollout=summary,
                )
                if not (
                    any(a > 0 for a in event["advantages"])
                    and any(a < 0 for a in event["advantages"])
                ):
                    event["status"] = "skipped_zero_advantage"
                else:
                    raw, metadata = assemble_training_data([group], advantages)
                    data, counts = recipe.prepare_data(raw)
                    if recovered:
                        # Session provenance is optional, raw observations/actions/logprobs are not.
                        data = [
                            replace(d, model_input_spans=None, loss_fn_input_spans=None)
                            for d in data
                        ]
                    event.update(counts, recovered_from_executed_group=recovered)
                    if (
                        state.get("training_input_tokens", 0) + counts["training_input_tokens"]
                        > config.max_training_tokens
                    ):
                        raise TokenLimitExceeded(
                            "Training input allowance exhausted; final evaluation retained"
                        )
                    state["training_input_tokens"] = (
                        state.get("training_input_tokens", 0) + counts["training_input_tokens"]
                    )
                    persist()
                    event["payload"] = recipe.save_training_payload(
                        out / f"training-payload-batch-{batch:04d}.json.gz",
                        raw,
                        data,
                        metadata,
                        summary["episodes"],
                    )
                    receipt = await recipe.apply_training_update(
                        client, data, config, control, event, journal
                    )
                    if not positive_gradient(receipt.metrics):
                        event["status"] = "service_confirmed_without_nonzero_gradient"
                        write_json(journal, event)
                        raise OutcomeUnknown(
                            "Optimizer acknowledged without usable gradient evidence; reconcile state"
                        )
                    event.update(
                        status="effective_update_confirmed_checkpoint_pending",
                        updates=state["updates"] + 1,
                    )
                    write_json(journal, event)
                    receipt_path = out / f"optimizer-receipt-batch-{batch:04d}.json"
                    write_json(receipt_path, {**event, "status": "effective_update_confirmed"})
                    current = await checkpoint(event["updates"], event, evidence_ref(receipt_path))
                    state["updates"] = event["updates"]
                    state["current_checkpoint"] = evidence_ref(
                        out / f"checkpoint-{state['updates']:04d}.json"
                    )
                    event.update(
                        status="effective_update_confirmed", checkpoint=state["current_checkpoint"]
                    )
                    write_json(journal, event)
                    persist()
                    sampler = await sampler_for(current)
                state["completed_batches"] += 1
                state["batches"].append(event)
                write_json(journal, event)
                persist()
                render_run_report(out, state)
            except TokenLimitExceeded as exc:
                state["stop_reason"] = str(exc)
                state["incomplete_batch"] = {**event, "status": "stopped_token_limit"}
                persist()
                break
            if (
                config.evaluation_enabled
                and state["completed_batches"] % config.evaluation_interval == 0
            ):
                try:
                    await evaluate_batch(state["completed_batches"])
                except TokenLimitExceeded as exc:
                    state["stop_reason"] = "Process evaluation token limit reached: " + str(exc)
                    persist()
                    break
        # Model choice is fixed before any final outcome is examined.
        state["final_selection"] = {
            "rule": "last_effective_update",
            "base": state["base_checkpoint"],
            "final": state["current_checkpoint"],
            "updates": state["updates"],
            "seed": config.final_seed,
            "n_per_model": config.final_eval_episodes if config.evaluation_enabled else 0,
            "evaluation_enabled": config.evaluation_enabled,
        }
        persist()
        for name in ("base", "final") if config.evaluation_enabled else ():
            if name in state["final_evaluations"]:
                continue
            saved = read_evidence(state["final_selection"][name])
            final_sampler = await sampler_for(saved)
            _, summary, _ = await cohort(
                f"final-{name}",
                config.final_eval_episodes,
                config.final_seed,
                final_sampler,
                policy_ref=state["final_selection"][name],
            )
            state["final_evaluations"][name] = {
                **summary,
                "checkpoint": state["final_selection"][name],
            }
            persist()
            render_run_report(out, state)
        state["status"] = (
            "complete" if state["completed_batches"] == config.batches else stopped_status
        )
        state["phase"] = "complete"
        status = "success"
        if recipe.fingerprint(config, task) != inputs:
            raise IntegrityError("Frozen inputs changed during experiment")
        persist()
        render_run_report(out, state)
        return state
    except asyncio.CancelledError:
        state["status"] = "interrupted"
        raise
    except BaseException as exc:
        state.update(
            status="needs_reconciliation" if isinstance(exc, OutcomeUnknown) else "failed",
            exception=type(exc).__name__,
            error=str(exc)[:1000],
        )
        raise
    finally:
        try:
            persist()
            render_run_report(out, state)
        finally:
            await service.close(status)
