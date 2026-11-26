"""Prepare one bounded retry of an entirely ungraded infrastructure-failed group.

No network calls or model updates are made here. The original controller then
resumes its latest confirmed checkpoint, with unchanged inputs and token counters.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.batch_recovery import cleanup_refs
from elt_rl.errors import IntegrityError
from elt_rl.lineage import evidence_ref, read_evidence
from elt_rl.task_scheduling import expected_task_fingerprint


def validate_ungraded_episode(raw, *, label, index, known, inputs):
    report = raw.get("report")
    if (
        raw.get("format") != "elt_episode_v4"
        or raw.get("episode_id") != f"{label}-{index:03d}"
        or raw.get("fingerprint") != expected_task_fingerprint(inputs, label)
        or raw.get("implementation_sha256") != inputs["implementation_sha256"]
        or raw.get("policy_checkpoint") != known["checkpoint"]
        or raw.get("termination")
        not in {"infrastructure_CancelledError", "infrastructure_InfrastructureError"}
        or raw.get("execution_result") is not None
        or (
            report is not None
            and (
                report.get("failure_type") != "infrastructure"
                or report.get("training_eligible") is not False
                or report.get("training_reward") is not None
                or report.get("reward_result") is not None
            )
        )
    ):
        raise IntegrityError("Retry requires unchanged, entirely ungraded infrastructure slots")


def _finish_moves(moves):
    for move in moves:
        source, target = Path(move["source"]), Path(move["target"])
        if source.exists() and target.exists():
            raise IntegrityError("Retry archive and original both exist")
        if source.exists():
            source.rename(target)
        if not target.exists() or target.is_symlink():
            raise IntegrityError("Infrastructure attempt archive is missing")
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() != move["sha256"]:
            raise IntegrityError("Infrastructure attempt archive changed")


def prepare_retry(out):
    out = Path(out).resolve(strict=True)
    with (out / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / "protocol-state.json"
        state = json.loads(state_path.read_text())
        batch = state["completed_batches"] + 1
        label = f"train-batch-{batch:04d}"
        known = state.get("cohorts", {}).get(label, {})
        journal = out / f"{label}-infrastructure-retry.json"
        if journal.exists():
            record = json.loads(journal.read_text())
            if record["status"] == "ready":
                if known.get("status") == "retry_prepared":
                    read_evidence(known["infrastructure_retry"])
                    return record
                if evidence_ref(state_path) != record["original_state"]:
                    raise IntegrityError("One infrastructure retry was already consumed")
            read_evidence(record["state_before"])
            if evidence_ref(state_path) != record["original_state"]:
                raise IntegrityError("State changed during infrastructure retry preparation")
        else:
            from elt_rl.train import TrainConfig, checked_checkpoint, fingerprint, preflight

            if (
                state.get("status") != "failed"
                or state.get("exception") != "InfrastructureError"
                or state.get("phase") != label
                or state.get("attempted_batches") != batch
                or known.get("status") != "dispatching"
                or known.get("checkpoint") != state.get("current_checkpoint")
                or known.get("infrastructure_retry") is not None
                or (out / f"optimization-batch-{batch:04d}.json").exists()
                or (out / f"training-payload-batch-{batch:04d}.json.gz").exists()
                or (out / f"{label}.json").exists()
            ):
                raise IntegrityError("Retry requires a failed cohort before any optimizer dispatch")
            config = TrainConfig(**state["config"])
            inputs = state["fingerprint"]
            if fingerprint(config, preflight(config)) != inputs:
                raise IntegrityError("Frozen implementation or execution inputs changed")
            checkpoint = checked_checkpoint(known["checkpoint"]["artifact_path"], config, inputs)
            read_evidence(known["checkpoint"])
            if checkpoint["updates"] != state["updates"]:
                raise IntegrityError("Retry checkpoint is not the latest confirmed update")
            if (
                known["size"] != config.group_size
                or known["seed"] != (config.seed + 1_000_000 + (batch - 1) * 100_000) % 2**31
                or known.get("sampler_path") not in (None, checkpoint["sampler_path"])
            ):
                raise IntegrityError("Retry would change the group or sampling policy")
            entries, infrastructure_failures = [], 0
            for index in range(known["size"]):
                path = out / "episodes" / f"{label}-{index:03d}.json"
                if path.is_symlink():
                    raise IntegrityError("Episode cannot be a symlink")
                raw = json.loads(path.read_text())
                validate_ungraded_episode(raw, label=label, index=index, known=known, inputs=inputs)
                infrastructure_failures += (
                    raw["termination"] == "infrastructure_InfrastructureError"
                )
                entries.append({"artifact": evidence_ref(path), "cleanup_refs": cleanup_refs(path)})
            if not infrastructure_failures:
                raise IntegrityError("Manual cancellation is not an infrastructure failure")
            archive = out / "infrastructure-training-attempts" / label / "attempt-0001"
            archive.mkdir(parents=True, mode=0o700, exist_ok=False)
            original = evidence_ref(state_path)
            backup = archive / "protocol-state-before.json"
            shutil.copyfile(state_path, backup)
            backup.chmod(0o600)
            moves = []
            for entry in entries:
                path = Path(entry["artifact"]["artifact_path"])
                moves.append(
                    {
                        "source": str(path),
                        "target": str(archive / path.name),
                        "sha256": entry["artifact"]["sha256"],
                    }
                )
                files = path.with_name(path.stem + "-files")
                if files.is_symlink():
                    raise IntegrityError("Episode files cannot be a symlink")
                if files.exists():
                    moves.append({"source": str(files), "target": str(archive / files.name)})
            record = {
                "format": "elt_infrastructure_training_retry_v1",
                "status": "prepared",
                "label": label,
                "retry_limit": 1,
                "retry_index": 1,
                "reason": state.get("error"),
                "checkpoint": known["checkpoint"],
                "original_state": original,
                "state_before": evidence_ref(backup),
                "original_cohort": known,
                "episodes": entries,
                "moves": moves,
                "sampling_counters_preserved": state["sampling"],
                "reward_bearing_episodes_discarded": 0,
                "optimizer_dispatched": False,
            }
            write_json(journal, record)
        _finish_moves(record["moves"])
        record["status"] = "ready"
        write_json(journal, record)
        state["cohorts"][label] = {
            **known,
            "status": "retry_prepared",
            "infrastructure_retry": evidence_ref(journal),
        }
        state.setdefault("recovery_events", []).append(
            {
                "kind": "ungraded_infrastructure_training_retry",
                "batch": batch,
                "retry": evidence_ref(journal),
            }
        )
        write_json(state_path, state)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = prepare_retry(args.run)
    print(
        json.dumps(
            {k: result[k] for k in ["status", "label", "retry_index", "optimizer_dispatched"]}
        )
    )
