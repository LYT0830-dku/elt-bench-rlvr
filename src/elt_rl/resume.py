"""Single-consumer checkpoint handoffs across resumed run directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from elt_rl.errors import IntegrityError


def canonical_checkpoint(path):
    """Latest aliases and named checkpoints share the same consumption record."""
    supplied = Path(path).resolve(strict=True)
    saved = json.loads(supplied.read_text())
    updates = saved.get("updates")
    steps = saved.get("config", {}).get("batches")
    if (
        type(steps) is not int
        or steps < 1
        or saved.get("fingerprint", {}).get("training_protocol", {}).get("batches") != steps
        or type(updates) is not int
        or not 0 <= updates <= steps
    ):
        raise IntegrityError("Invalid checkpoint update identity")
    canonical = supplied.parent / f"checkpoint-{updates:04d}.json"
    if canonical.is_symlink() or not canonical.is_file():
        raise IntegrityError("Resume requires its canonical named checkpoint")
    if canonical.read_bytes() != supplied.read_bytes():
        raise IntegrityError("Checkpoint alias differs from the canonical checkpoint")
    return canonical


def claim_path(checkpoint):
    return checkpoint.with_name(checkpoint.name + ".resume-claim.json")


def validate_resume_owner(path):
    """An unclaimed/rejected child copy cannot become a new resume ancestor."""
    checkpoint = canonical_checkpoint(path)
    config_path = checkpoint.parent / "run-config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise IntegrityError("Resume requires its original run configuration")
    config = json.loads(config_path.read_text())
    source = config.get("config", {}).get("resume_checkpoint")
    if source is None:
        return
    source = canonical_checkpoint(source)
    try:
        claim = json.loads(claim_path(source).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise IntegrityError("Resumed run has no confirmed handoff claim") from exc
    if (
        claim.get("status") != "claimed"
        or claim.get("destination") != str(checkpoint.parent.resolve())
        or claim.get("source_checkpoint") != str(source)
        or claim.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        raise IntegrityError("Checkpoint handoff belongs to a different resumed run")
