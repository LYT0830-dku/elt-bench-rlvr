"""Pinned official task-specific source and target manifest preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from elt_rl.benchmark import ORIGINAL_REVISION, UPSTREAM, task_material


def prepare_task(repo: Path, name: str, assets: Path, runtime_config: Path, output: Path) -> Path:
    """Create an original task pack. Source bytes only initialize real source services."""
    files, lock = task_material(Path(repo), name)
    if name != "cars":
        raise ValueError(
            "Author a task pack with registered source adapters for another original task; the RL loop is unchanged"
        )
    if json.loads(Path(runtime_config).read_text()).get("execution_backend") != "native":
        raise ValueError("Task preparation requires the native execution backend")
    assets = Path(assets).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = lock["models"]
    target = spec[0]
    columns = [c["name"].lower() for c in target["columns"]]
    sources = []
    kinds = {"data": "rest", "price": "postgres", "country": "s3", "production": "csv_http"}
    for source, kind in kinds.items():
        schema = list(
            __import__("csv").DictReader(
                __import__("io").StringIO(files[f"elt-bench/{name}/schemas/{source}.csv"].decode())
            )
        )
        source_columns = [r["column_name"].lower() for r in schema]
        path = assets / "sources" / (source + (".jsonl" if source == "country" else ".csv"))
        if not path.is_file():
            raise ValueError(f"Missing original source asset: {source}")
        options = {"path": str(path), "format": "jsonl" if source == "country" else "csv"}
        if kind == "postgres":
            options["initial_types"] = {"id": "SERIAL", "price": "REAL"}
            options["initial_primary_keys"] = ["id"]
        if kind == "s3":
            options["key"] = "country.jsonl"
        if kind == "rest":
            options.update(
                url_path="/cars/data",
                numeric=[
                    "id",
                    "mpg",
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "model",
                ],
            )
        if kind == "csv_http":
            options["url_path"] = "/cars/production.csv"
        sources.append(
            {"name": source, "kind": kind, "columns": source_columns, "options": options}
        )
    ref = assets / "references/countries.csv"
    reference_provenance = json.loads((assets / "reference-provenance.json").read_text())
    source_provenance = json.loads((assets / "source-provenance.json").read_text())
    data = {
        "name": name,
        "mode": "native",
        "description": "Complete the original ELT-Bench task by configuring and executing native extraction/loading and dbt transformations.",
        "benchmark_repo": str(Path(repo).resolve()),
        "benchmark_lock": {
            k: lock[k] for k in ["repository", "revision", "task", "files_sha256", "source_counts"]
        },
        "public_config": {
            "original_config": "task/original_config.yaml",
            "connection_facts": "provided at episode initialization",
            "credentials": "host-managed source and warehouse capabilities",
            "connector_decisions": "model selects source, table, columns, destination and loading mode",
        },
        "reference_provenance": reference_provenance,
        "source_provenance": source_provenance,
        "source_sha256": {
            s["name"]: hashlib.sha256(Path(s["options"]["path"]).read_bytes()).hexdigest()
            for s in sources
        },
        "reference_sha256": {"countries": hashlib.sha256(ref.read_bytes()).hexdigest()},
        "sources": sources,
        "targets": [
            {
                "name": "countries",
                "columns": columns,
                "keys": ["origin"],
                "numeric": [
                    "origin",
                    "num_cars",
                    "num_car_worth_over_35000_and_acceleration_14",
                    "avg_car_price",
                ],
                "expected": str(ref),
                "descriptions": {c["name"].lower(): c["description"] for c in target["columns"]},
            }
        ],
        "runtime": {
            "config_file": str(Path(runtime_config).resolve()),
            "benchmark_repo": str(Path(repo).resolve()),
        },
        "provenance": {
            "benchmark_repository": UPSTREAM,
            "benchmark_revision": ORIGINAL_REVISION,
            "workflow": "original_native",
            "task_definition": "original public specification and original-linked ground truth",
        },
    }
    if json.loads(Path(runtime_config).read_text()).get("execution_backend") == "native":
        data["description"] = (
            "Complete the original ELT-Bench task: author loading configuration, perform real extraction/loading, author and run dbt, repair errors, and submit a replayable pipeline. See task/native_loading.md for the installed tools."
        )
        data["public_config"]["credentials"] = "host-managed source and warehouse capabilities"
        data["public_config"]["connector_decisions"] = (
            "model authors source/connector/stream/columns/types, assigned destination tables and loading mode in elt/loading.json"
        )
        data["provenance"]["execution_toolchain"] = (
            "native_loading_v1; adapted from Airbyte/Terraform, not original platform execution"
        )
    path = output / "task.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
