"""Checkpoint lineage verified from immutable optimizer receipts and curve snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from elt_rl.errors import IntegrityError


def evidence_ref(path):
    path = Path(path).resolve()
    return {"artifact_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_evidence(ref):
    path = Path(ref["artifact_path"])
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise IntegrityError("Evidence link digest mismatch")
    return json.loads(path.read_text())


def positive_gradient(metrics):
    value = (metrics or {}).get("unclipped_grad_l2:mean")
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def verify_lineage(saved):
    if not saved.get("resume_allowed", False):
        raise IntegrityError("Checkpoint requires reconciliation before use")
    receipt = read_evidence(saved["save_receipt"])
    if (
        receipt.get("status") != "confirmed"
        or receipt.get("updates") != saved["updates"]
        or receipt.get("state_result", {}).get("path") != saved["path"]
        or receipt.get("sampler_result", {}).get("path") != saved["sampler_path"]
    ):
        raise IntegrityError("Checkpoint state/sampler do not match the save service receipts")
    lineage = saved.get("lineage")
    if not isinstance(lineage, dict):
        raise IntegrityError("Missing checkpoint lineage")
    updates = saved["updates"]
    if updates:
        receipt = read_evidence(lineage["optimizer_receipt"])
        if (
            receipt.get("status") != "effective_update_confirmed"
            or receipt.get("updates") != updates
            or not positive_gradient(receipt.get("optimizer_metrics"))
        ):
            raise IntegrityError("Checkpoint has no confirmed nonzero-gradient optimizer receipt")
        parent = read_evidence(lineage["parent_checkpoint"])
        if parent.get("updates") != updates - 1 or parent.get("fingerprint") != saved.get(
            "fingerprint"
        ):
            raise IntegrityError("Checkpoint lineage is discontinuous")
        verify_lineage(parent)
    elif lineage.get("optimizer_receipt") or lineage.get("parent_checkpoint"):
        raise IntegrityError("Initial checkpoint cannot claim an optimizer update")
    nodes = read_evidence(lineage["curve_snapshot"]) if lineage.get("curve_snapshot") else []
    if lineage.get("schedule") == "batches_v1":
        batches = [node.get("batch") for node in nodes]
        interval = saved["config"]["evaluation_interval"]
        if batches != list(range(0, interval * len(nodes), interval)) or any(
            type(node.get("updates")) is not int or not 0 <= node["updates"] <= updates
            for node in nodes
        ):
            raise IntegrityError("Checkpoint process evaluation schedule is inconsistent")
    else:
        expected = updates if lineage.get("evaluation_pending", True) else updates + 1
        if [node["step"] for node in nodes] != list(range(expected)):
            raise IntegrityError("Checkpoint curve omits or duplicates an evaluation step")
    protocol = saved["fingerprint"]["evaluation_protocol"]
    for node in nodes:
        if node["n"] != protocol["n"] or node["seed"] != protocol["seed"]:
            raise IntegrityError("Checkpoint evaluation protocol changed")
    return nodes
