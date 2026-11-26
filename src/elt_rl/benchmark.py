"""Original ELT-Bench task material, read from a verified upstream Git tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from elt_rl.errors import IntegrityError

ORIGINAL_REVISION = "e2e377766604f61930165f20d4fcab8400a3e997"
UPSTREAM = "https://github.com/uiuc-kang-lab/ELT-Bench"


def blob(repo: Path, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{ORIGINAL_REVISION}:{path}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise IntegrityError(f"Missing original benchmark file: {path}") from exc


def task_material(repo: Path, name: str) -> tuple[dict[str, bytes], dict]:
    from elt_rl.task import identifier

    name = identifier(name)
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != ORIGINAL_REVISION:
        raise IntegrityError(f"Original ELT-Bench requires {ORIGINAL_REVISION}, got {actual}")
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ORIGINAL_REVISION], text=True
    ).splitlines()
    required = [
        p for p in paths if p.startswith(f"elt-bench/{name}/") or p.startswith("documentation/")
    ]
    required += [
        "agents/SWE-agent/config/elt.yaml",
        "setup/main.tf",
        "setup/check_job_status.py",
        "evaluation/eva.py",
        "evaluation/eva_stage1.py",
        "evaluation/eva_stage2.py",
        "evaluation/table.json",
        "setup/elt_bench.yaml",
        "elt-docker/rest_api/app.py",
        "setup/elt_setup.sh",
    ]
    required += [p for p in paths if p.startswith(f"setup/sources/{name}/")]
    required += [p for p in paths if p.startswith(f"evaluation/{name}/")]
    files = {p: blob(repo, p) for p in required}
    if f"elt-bench/{name}/config.yaml" not in files:
        raise ValueError(f"Unknown original task: {name}")
    for p, content in files.items():
        if not (repo / p).is_file() or (repo / p).read_bytes() != content:
            raise IntegrityError(f"Modified original benchmark file: {p}")
    config = yaml.safe_load(files[f"elt-bench/{name}/config.yaml"])
    models = yaml.safe_load(files[f"elt-bench/{name}/data_model.yaml"])["models"]
    lock = {
        "repository": UPSTREAM,
        "revision": ORIGINAL_REVISION,
        "task": name,
        "files_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in files.items()},
        "source_counts": json.loads(files["evaluation/table.json"])[name],
        "models": models,
        "config": config,
    }
    return files, lock


def public_project(files: dict[str, bytes], name: str, public_config: dict) -> dict[str, str]:
    """Keep original specifications/docs; substitute host connection facts, not solutions."""
    project = {}
    for path, content in files.items():
        if path.startswith(f"elt-bench/{name}/"):
            project["task/" + path.split(f"elt-bench/{name}/", 1)[1]] = content.decode()
        elif path.startswith("documentation/"):
            project["task/" + path] = content.decode()
    project["task/original_config.yaml"] = project["task/config.yaml"]
    project["task/config.yaml"] = yaml.safe_dump(public_config, sort_keys=False)
    project["task/agent_prompt.yaml"] = files["agents/SWE-agent/config/elt.yaml"].decode()
    project["elt/main.tf"] = files["setup/main.tf"].decode()
    return project


def prepare_task(repo: Path, name: str, assets: Path, runtime_config: Path, output: Path) -> Path:
    from elt_rl.task_adapters import TASK_PREPARERS

    return TASK_PREPARERS.get(name)(repo, name, assets, runtime_config, output)
