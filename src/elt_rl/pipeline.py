"""Execute a saved native loading and dbt submission through the environment tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from elt_rl.environment import ELTEnvironment
from elt_rl.project import MAX_FILE_BYTES, MAX_PROJECT_BYTES, project_path
from elt_rl.task import Task


def load_pipeline(path: Path) -> dict:
    """Verify a submitted manifest and read the files the model actually wrote."""
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != "elt_project_v1":
        raise ValueError("Expected an elt_project_v1 submission manifest")
    files, recipe = manifest.get("files"), manifest.get("recipe")
    if not isinstance(files, dict) or not files or not isinstance(recipe, dict):
        raise ValueError("Submission requires project files and a sync recipe")
    connections = recipe.get("sync_connections")
    if (
        recipe.get("dbt_command") != "run"
        or not isinstance(connections, list)
        or not connections
        or any(not isinstance(name, str) or not name for name in connections)
        or len(connections) > 200
    ):
        raise ValueError("Invalid submission sync or dbt recipe")
    root = path.resolve().parent
    contents = {}
    total = 0
    for name, digest in files.items():
        file = project_path(root, name)
        if not name.startswith(("elt/", "dbt/")) or not file.is_file():
            raise ValueError(f"Missing submission file: {name}")
        size = file.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_PROJECT_BYTES:
            raise ValueError("Submitted project exceeds its byte budget")
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Submission file digest differs: {name}")
        contents[name] = data.decode("utf-8")
    return {
        "files": contents,
        "sync_connections": connections,
        **({"loading_backend": recipe["loading_backend"]} if recipe.get("loading_backend") else {}),
    }


def pipeline_actions(pipeline: dict) -> list[dict]:
    actions = [
        {"tool": "write_file", "path": path, "content": content}
        for path, content in pipeline["files"].items()
    ]
    if pipeline.get("loading_backend") != "native_loading_v1":
        raise ValueError("Pipeline must use native_loading_v1")
    actions.extend(
        {"tool": "loading", "command": command} for command in ("validate", "plan", "apply")
    )
    # Each wait uses the ID returned by its preceding sync.
    for connection in pipeline["sync_connections"]:
        actions.append({"tool": "sync", "connection": connection})
        actions.append({"tool": "wait_sync", "job_id": "$last_job"})
    actions.extend(
        [
            {"tool": "dbt", "command": "compile"},
            {"tool": "dbt", "command": "run"},
            {"tool": "submit"},
        ]
    )
    return actions


def run_pipeline(
    task: Task,
    pipeline: dict,
    destination: str = "duckdb",
    *,
    artifact_path: Path | None = None,
    allow_remote=False,
) -> dict:
    with ELTEnvironment(
        task,
        destination,
        artifact_path=artifact_path,
        allow_remote=allow_remote,
    ) as env:
        last_job = None
        env.initial()
        for action in pipeline_actions(pipeline):
            if action.get("job_id") == "$last_job":
                action = {**action, "job_id": last_job}
            step = env.step(action)
            if action["tool"] == "sync":
                last_job = step.observation.get("job_id")
            if "error" in step.observation:
                env.done = True
                env.termination = "pipeline_error"
            if env.done:
                break
    return env.artifact()
