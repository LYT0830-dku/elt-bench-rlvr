"""Crash-safe partial evaluation recovery in the original run and sampling records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from elt_rl.artifacts import valid_submission_receipt, write_json
from elt_rl.errors import IntegrityError
from elt_rl.lineage import evidence_ref, read_evidence, verify_lineage


def cleanup_refs(path):
    """Require terminal ownership receipts for every allocated native resource."""
    root = Path(path).with_name(Path(path).stem + "-files")
    refs = []
    for file in sorted(root.rglob("*.json")):
        if file.name == "runtime.json" and file.parent.name == "exploration":
            data = json.loads(file.read_text())
            ends = [e for e in data.get("events", []) if e.get("kind") == "cleanup"]
            warehouses = [e for e in data.get("events", []) if e.get("kind") == "warehouse"]
            local = warehouses and warehouses[-1].get("backend") == "duckdb"
            local_closed = (
                local
                and warehouses[-1].get("state") == "closed"
                and warehouses[-1].get("active_connections") == 0
            )
            if (
                not ends
                or ends[-1].get("status") != "closed"
                or ends[-1].get("failures") != []
                or not warehouses
                or (
                    not local_closed
                    and (
                        warehouses[-1].get("state") != "closed"
                        or warehouses[-1].get("remaining_schemas") != []
                        or warehouses[-1].get("role_exists") is not False
                        or warehouses[-1].get("schema_creation_pending") != []
                        or warehouses[-1].get("role_creation_pending") is not False
                    )
                )
            ):
                raise IntegrityError("Partial evaluation has unresolved warehouse cleanup")
            refs.append(evidence_ref(file))
        elif file.name == "resources.json" and file.parent.name == "sources":
            data = json.loads(file.read_text())
            if (
                data.get("backend") != "native_services_v1"
                or data.get("state") != "closed"
                or data.get("listeners") != []
                or any(p.get("running") is not False for p in data.get("processes", []))
            ):
                raise IntegrityError("Partial evaluation has unresolved source cleanup")
            refs.append(evidence_ref(file))
    if not any(Path(r["artifact_path"]).name == "runtime.json" for r in refs):
        # A process killed before first tool can have no runtime, but no allocated resources either.
        raw = json.loads(Path(path).read_text())
        if raw.get("runtime") is not None:
            raise IntegrityError("Partial evaluation is missing cleanup receipts")
    return refs


def validate_episode(path, *, index, label, seed, inputs, config):
    import math

    from elt_rl.environment import REWARD_VERSION

    path = Path(path)
    if path.is_symlink():
        raise IntegrityError("Evaluation artifact cannot be a symlink")
    raw, artifact = json.loads(path.read_text()), evidence_ref(path)
    refs = cleanup_refs(path)
    report = raw.get("report") or {}
    from elt_rl.task_scheduling import expected_task_fingerprint

    if (
        raw.get("format") != "elt_episode_v4"
        or raw.get("mode") != expected_task_fingerprint(inputs, label).get("mode", "native")
        or raw.get("episode_id") != f"{label}-{index:03d}"
        or raw.get("fingerprint") != expected_task_fingerprint(inputs, label)
        or raw.get("destination") != config.destination
    ):
        raise IntegrityError("Partial evaluation task or episode identity changed")
    if raw.get("termination") == "infrastructure_CancelledError" and raw.get("done") is True:
        # Grading may finish before cancellation of its async result delivery.
        # Keep that actual report in the cancelled archive; never turn it into zero.
        if not valid_submission_receipt(raw) or report.get("training_eligible") is not True:
            raise IntegrityError("Cancelled completed submission has no valid grading receipt")
        if inputs.get("reward_configuration") is not None:
            from elt_rl.execution_scoring import verify_scoring_receipt

            verify_scoring_receipt(raw, inputs["reward_configuration"])
        return None
    if raw.get("done") is not True:
        if (
            raw.get("report") is not None
            or raw.get("termination") != "infrastructure_CancelledError"
        ):
            raise IntegrityError(
                "Only confirmed manually cancelled evaluation slots can be retried"
            )
        return None
    if (
        raw.get("termination") != "submitted"
        or not valid_submission_receipt(raw)
        or report.get("training_eligible") is not True
        or report.get("reward_version") != inputs.get("reward_version", REWARD_VERSION)
    ):
        raise IntegrityError("Completed evaluation has no valid direct grading receipt")
    if raw.get("implementation_sha256") != inputs["implementation_sha256"]:
        raise IntegrityError("Evaluation source differs from the recorded run")
    if inputs.get("reward_configuration") is not None:
        from elt_rl.execution_scoring import verify_scoring_receipt

        verify_scoring_receipt(raw, inputs["reward_configuration"])
    samples = raw.get("sampling", [])
    if not samples or raw.get("turns") != len(samples):
        raise IntegrityError("Evaluation sampling receipts are incomplete")
    for turn, event in enumerate(samples):
        tokens, probs = event.get("tokens"), event.get("logprobs")
        if (
            event.get("status") != "sampled"
            or event.get("seed") != (seed + 1009 * index + turn) % 2**31
            or any(event.get(k) != getattr(config, k) for k in ("temperature", "top_p", "top_k"))
            or not isinstance(tokens, list)
            or not tokens
            or not all(type(t) is int and t >= 0 for t in tokens)
            or not isinstance(probs, list)
            or len(probs) != len(tokens)
            or not all(type(p) in (int, float) and math.isfinite(p) for p in probs)
            or event.get("output_tokens") != len(tokens)
            or not event.get("sequence_id")
        ):
            raise IntegrityError(
                "Evaluation seed, sampling configuration or token evidence changed"
            )
    return {"artifact": artifact, "raw": raw, "cleanup_refs": refs}


def prepare_partial(out, known, *, label, size, seed, inputs, config):
    """Journal moves before mutation; repeated recovery continues the same archive."""
    if not label.startswith(("eval-batch-", "final-")):
        raise IntegrityError("Partial training cohorts cannot be automatically resampled")
    checkpoint = read_evidence(known["checkpoint"])
    verify_lineage(checkpoint)
    if (
        known.get("size") != size
        or known.get("seed") != seed
        or known.get("sampler_path") not in (None, checkpoint["sampler_path"])
    ):
        raise IntegrityError("Partial evaluation sampling policy or seed plan changed")
    journal = Path(out) / f"{label}-recovery.json"
    if journal.exists():
        recovery = json.loads(journal.read_text())
        if (
            recovery["checkpoint"] != known["checkpoint"]
            or recovery["label"] != label
            or recovery["size"] != size
            or recovery["seed"] != seed
        ):
            raise IntegrityError("Recovery journal belongs to different weights or slots")
        for entry in recovery["completed"].values():
            read_evidence(entry["artifact"])
            for ref in entry["cleanup_refs"]:
                read_evidence(ref)
    else:
        recovery = {
            "format": "elt_partial_evaluation_v1",
            "label": label,
            "size": size,
            "seed": seed,
            "checkpoint": known["checkpoint"],
            "completed": {},
            "archives": [],
        }
    # Finish any move interrupted between atomic rename and journal commit.
    for move in recovery["archives"]:
        source, target = Path(move["source"]), Path(move["target"])
        if source.exists() and not target.exists():
            source.rename(target)
        if not target.exists():
            raise IntegrityError("Cancelled evaluation archive is missing")
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() != move["sha256"]:
            raise IntegrityError("Cancelled evaluation archive was modified")
    cached = {}
    pending_moves = []
    for index in range(size):
        path = Path(out) / "episodes" / f"{label}-{index:03d}.json"
        if not path.exists():
            continue
        entry = validate_episode(
            path,
            index=index,
            label=label,
            seed=seed,
            inputs=inputs,
            config=config,
        )
        if entry is not None:
            cached[index] = entry
            recovery["completed"][str(index)] = {k: v for k, v in entry.items() if k != "raw"}
            continue
        attempt = sum(Path(m["source"]).name == path.name for m in recovery["archives"])
        target_root = (
            Path(out) / "cancelled-evaluation-attempts" / label / f"{index:03d}-{attempt:03d}"
        )
        target_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for source in (path, path.with_name(path.stem + "-files")):
            if not source.exists():
                continue
            move = {"source": str(source), "target": str(target_root / source.name)}
            if source.is_file():
                move["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            pending_moves.append(move)
    recovery["archives"].extend(pending_moves)
    write_json(journal, recovery)
    for move in pending_moves:
        Path(move["source"]).rename(move["target"])
        directory = os.open(Path(move["source"]).parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return cached, evidence_ref(journal)
