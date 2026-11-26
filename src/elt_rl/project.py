"""Guarded model workspace and immutable submitted project bundles."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

from elt_rl.artifacts import write_json
from elt_rl.errors import ActionError

MAX_FILE_BYTES = 200_000
MAX_PROJECT_BYTES = 4_000_000


def project_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ActionError("Invalid project path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(x in {".", ".."} or x.startswith(".") for x in rel.parts):
        raise ActionError("Use a relative public project path without hidden components")
    p = root.joinpath(*rel.parts)
    if not p.resolve().is_relative_to(root) or any(
        q.is_symlink() for q in [p, *p.parents] if q.is_relative_to(root)
    ):
        raise ActionError("Symlinks and paths outside the workspace are forbidden")
    return p


class Project:
    def __init__(self, root: Path, initial: dict[str, str]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.locked = {name for name in initial if name.startswith("task/")}
        for name, value in initial.items():
            p = self._path(name, write=False)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(value)
        self.submitted = False

    def _path(self, name: str, *, write: bool = False) -> Path:
        p = project_path(self.root, name)
        rel = p.relative_to(self.root)
        if write:
            from elt_rl.task import identifier

            try:
                for component in rel.parts[:-1]:
                    identifier(component)
                identifier(Path(rel.name).stem)
            except ValueError as exc:
                raise ActionError(
                    "Executable project paths must use plain SQL identifiers"
                ) from exc
            if self.submitted:
                raise ActionError("Submission is frozen")
            if name in self.locked or name.startswith("task/"):
                raise ActionError("Official task inputs are read-only")
            if not (
                (name.startswith("elt/") and name.endswith(".json"))
                or (name.startswith("dbt/") and name.endswith((".sql", ".yml", ".yaml")))
            ):
                raise ActionError("Write loading JSON in elt/ and dbt SQL/YAML in dbt/")
            if any(
                x in rel.parts
                for x in ["target", "logs", "dbt_packages", "macros", "seeds", "snapshots"]
            ):
                raise ActionError(
                    "Generated outputs, arbitrary macros and seed tables are not submission inputs"
                )
        return p

    def read(self, name: str) -> str:
        p = self._path(name)
        if not p.is_file():
            raise ActionError("File does not exist")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ActionError("File exceeds read budget")
        return p.read_text()

    def write(self, name: str, content: str):
        p = self._path(name, write=True)
        if not isinstance(content, str) or len(content.encode()) > MAX_FILE_BYTES:
            raise ActionError("File exceeds write budget")
        if (
            sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
            - (p.stat().st_size if p.is_file() else 0)
            + len(content.encode())
            > MAX_PROJECT_BYTES
        ):
            raise ActionError("Project exceeds byte budget")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"written": name, "sha256": hashlib.sha256(content.encode()).hexdigest()}

    def edit(self, name: str, old: str, new: str):
        self._path(name, write=True)
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ActionError("Edit requires nonempty old text and string new text")
        content = self.read(name)
        start = content.find(old)
        if start < 0 or content.find(old, start + 1) >= 0:
            raise ActionError("Old text must match exactly once")
        return self.write(name, content[:start] + new + content[start + len(old) :])

    def delete(self, name):
        path = self._path(name, write=True)
        if not path.is_file():
            raise ActionError("File does not exist")
        path.unlink()
        return {"deleted": name}

    def files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file()
            and not any(
                x.startswith(".") or x in {"target", "logs", "dbt_packages"}
                for x in p.relative_to(self.root).parts
            )
        )

    def execution_hashes(self, prefix: str) -> dict[str, str]:
        """Identity of model-authored inputs, excluding host credentials and outputs."""
        return {
            name: hashlib.sha256(self._path(name).read_bytes()).hexdigest()
            for name in self.files()
            if name.startswith(prefix + "/")
        }

    def freeze(self, output: Path, recipe: dict) -> dict:
        if self.submitted:
            raise ActionError("Already submitted")
        names = [n for n in self.files() if n.startswith(("elt/", "dbt/"))]
        if (
            "elt/loading.json" not in names
            or "dbt/dbt_project.yml" not in names
            or not any(n.startswith("dbt/models/") and n.endswith(".sql") for n in names)
        ):
            raise ActionError("Submit loading configuration, dbt_project.yml and dbt model SQL")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name in names:
            data = self._path(name).read_bytes()
            p = output / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.chmod(p, 0o444)
            hashes[name] = hashlib.sha256(data).hexdigest()
        manifest = {"format": "elt_project_v1", "files": hashes, "recipe": recipe}
        write_json(output / "pipeline.json", manifest)
        self.submitted = True
        return manifest
