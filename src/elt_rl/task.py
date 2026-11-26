"""One task contract for local integration and the original ELT workflow."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from elt_rl.errors import IntegrityError


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value.lower()


def column_names(value, field_name="columns"):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be nonempty")
    names = tuple(identifier(x) for x in value)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate {field_name}")
    return names


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    columns: tuple[str, ...]
    options: dict = field(default_factory=dict)
    required: bool = True


@dataclass(frozen=True)
class Target:
    name: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]
    numeric: tuple[str, ...]
    expected: Path
    descriptions: dict = field(default_factory=dict)
    atol: float = 1e-8
    rtol: float = 1e-6


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    sources: tuple[Source, ...]
    targets: tuple[Target, ...]
    mode: str
    project: dict[str, str]
    runtime: dict
    provenance: dict = field(default_factory=dict)
    manifest: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> Task:
        p = Path(path).resolve()
        d = yaml.safe_load(p.read_text())
        if not isinstance(d, dict):
            raise ValueError("Task manifest must be a mapping")
        allowed = {
            "name",
            "mode",
            "description",
            "sources",
            "targets",
            "project_dir",
            "runtime",
            "provenance",
            "benchmark_repo",
            "benchmark_lock",
            "public_config",
            "reference_provenance",
            "source_provenance",
            "source_sha256",
            "reference_sha256",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ValueError("Unknown task settings: " + ", ".join(sorted(map(str, unknown))))
        if d.get("mode") not in {"local_test", "native"}:
            raise ValueError("Task mode must be local_test or native")
        name = identifier(d["name"])
        sources = []
        targets = []
        for s in d["sources"]:
            opts = dict(s.get("options", {}))
            if "path" in opts:
                opts["path"] = str((p.parent / opts["path"]).resolve())
            sources.append(
                Source(
                    identifier(s["name"]),
                    s["kind"],
                    column_names(s["columns"]),
                    opts,
                    s.get("required", True),
                )
            )
        for t in d["targets"]:
            ref = (p.parent / t["expected"]).resolve()
            cols = column_names(t["columns"])
            keys = column_names(t["keys"], "keys")
            if not set(keys) <= set(cols):
                raise ValueError("Target keys absent from columns")
            targets.append(
                Target(
                    identifier(t["name"]),
                    cols,
                    keys,
                    tuple(t.get("numeric", [])),
                    ref,
                    t.get("descriptions", {}),
                    float(t.get("atol", 1e-8)),
                    float(t.get("rtol", 1e-6)),
                )
            )
        project = {}
        if d["mode"] == "native":
            from elt_rl.benchmark import public_project, task_material

            repo = (p.parent / d["benchmark_repo"]).resolve()
            files, lock = task_material(repo, name)
            if d.get("benchmark_lock") != {
                k: lock[k]
                for k in ["repository", "revision", "task", "files_sha256", "source_counts"]
            }:
                raise IntegrityError(
                    "Native task benchmark lock is missing or disagrees with upstream"
                )
            if not d.get("reference_provenance", {}).get("files"):
                raise IntegrityError("Native reference provenance is required")
            if set(s.name for s in sources) != set(lock["source_counts"]) or any(
                not s.required for s in sources
            ):
                raise IntegrityError("Native task must include every original required source")
            for source in sources:
                data = Path(source.options["path"]).read_bytes()
                if d.get("source_sha256", {}).get(source.name) != hashlib.sha256(data).hexdigest():
                    raise IntegrityError("Original source digest mismatch")
            original_targets = {
                m["name"]: {c["name"].lower() for c in m["columns"]} for m in lock["models"]
            }
            if {t.name: set(t.columns) for t in targets} != original_targets:
                raise IntegrityError("Native target contract differs from original data model")
            import io

            for source in sources:
                specification = list(
                    csv.DictReader(
                        io.StringIO(files[f"elt-bench/{name}/schemas/{source.name}.csv"].decode())
                    )
                )
                if set(source.columns) != {
                    column["column_name"].lower() for column in specification
                }:
                    raise IntegrityError("Native source columns differ from the original schema")
                raw = Path(source.options["path"]).read_text()
                count = (
                    sum(bool(line.strip()) for line in raw.splitlines())
                    if source.options.get("format") == "jsonl"
                    else sum(1 for _ in csv.DictReader(io.StringIO(raw)))
                )
                if count != lock["source_counts"][source.name]:
                    raise IntegrityError("Original source cardinality mismatch")
            project = public_project(files, name, d["public_config"])
        else:
            project_dir = (p.parent / d["project_dir"]).resolve()
            for f in sorted(project_dir.rglob("*")):
                if f.is_symlink():
                    raise IntegrityError("Task project cannot contain symlinks")
                if f.is_file():
                    project[str(f.relative_to(project_dir))] = f.read_text()
        runtime = dict(d.get("runtime", {}))
        for k in ["config_file", "source_manifest", "benchmark_repo"]:
            if k in runtime:
                runtime[k] = str((p.parent / runtime[k]).resolve())
        if d["mode"] == "native":
            runtime["benchmark_repo"] = str(repo)
        import json

        if (
            runtime.get("config_file")
            and json.loads(Path(runtime["config_file"]).read_text()).get("execution_backend")
            == "native"
        ):
            from elt_rl.native_protocol import GUIDE

            project.pop("elt/main.tf", None)
            project["task/native_loading.md"] = GUIDE
        task = cls(
            name,
            d["description"],
            tuple(sources),
            tuple(targets),
            d["mode"],
            project,
            runtime,
            d.get("provenance", {}),
            p,
        )
        task.verify_references(d.get("reference_sha256", {}))
        from elt_rl.source_data import source_rows

        source_rows(task)
        return task

    def verify_references(self, expected_hashes: dict):
        for target in self.targets:
            data = target.expected.read_bytes()
            if expected_hashes.get(target.name) != hashlib.sha256(data).hexdigest():
                raise IntegrityError(f"Missing/mismatched reference digest: {target.name}")
            with target.expected.open(newline="") as f:
                fields = next(csv.reader(f))
            if set(x.lower() for x in fields) != set(target.columns):
                raise IntegrityError(f"Invalid reference schema: {target.name}")

    def public(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "workflow": self.mode,
            "project_files": sorted(self.project),
            "sources": [
                {"name": s.name, "kind": s.kind, "columns": s.columns} for s in self.sources
            ],
            "targets": [
                {"name": t.name, "columns": t.columns, "descriptions": t.descriptions}
                for t in self.targets
            ],
        }

    def fingerprint(self) -> dict:
        def digest(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        return {
            "manifest": digest(self.manifest) if self.manifest else None,
            "mode": self.mode,
            "project": {n: hashlib.sha256(v.encode()).hexdigest() for n, v in self.project.items()},
            "references": {t.name: digest(t.expected) for t in self.targets},
            "provenance": self.provenance,
            "source_files": {
                s.name: digest(s.options["path"]) for s in self.sources if "path" in s.options
            },
            "runtime_config": {
                k: digest(v)
                for k, v in self.runtime.items()
                if k.endswith("_file") and Path(v).is_file()
            },
        }
