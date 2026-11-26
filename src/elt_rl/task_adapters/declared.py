"""Manifest preparation for reviewed, declarative official task packs."""

import csv
import hashlib
import io
import json
from pathlib import Path

import yaml

from elt_rl.benchmark import ORIGINAL_REVISION, UPSTREAM, task_material


def prepare_declared(repo, name, assets, runtime_config, output):
    from elt_rl.task_adapters import pack_definitions

    definition = pack_definitions()[name]
    files, lock = task_material(Path(repo), name)
    assets, output = Path(assets).resolve(), Path(output).resolve()
    sources = []
    for source in definition["sources"]:
        source = json.loads(json.dumps(source))
        schema = csv.DictReader(
            io.StringIO(files[f"elt-bench/{name}/schemas/{source['name']}.csv"].decode())
        )
        source["columns"] = [row["column_name"].lower() for row in schema]
        source["options"]["path"] = str(assets / source["options"]["path"])
        sources.append(source)
    targets = []
    for target in lock["models"]:
        declaration = definition["targets"][target["name"]]
        targets.append(
            {
                "name": target["name"],
                "columns": [column["name"].lower() for column in target["columns"]],
                "keys": declaration["keys"],
                "numeric": declaration["numeric"],
                "expected": str(assets / "references" / (target["name"] + ".csv")),
                "descriptions": {
                    column["name"].lower(): column["description"] for column in target["columns"]
                },
            }
        )
    data = {
        "name": name,
        "mode": "native",
        "description": "Complete the original ELT-Bench task: author loading configuration, perform real extraction/loading, author and run dbt, repair errors, and submit a replayable pipeline. See task/native_loading.md for the installed tools.",
        "benchmark_repo": str(Path(repo).resolve()),
        "benchmark_lock": {
            key: lock[key]
            for key in ("repository", "revision", "task", "files_sha256", "source_counts")
        },
        "public_config": {
            "original_config": "task/original_config.yaml",
            "connection_facts": "provided at episode initialization",
            "credentials": "host-managed source and warehouse capabilities",
            "connector_decisions": "model authors source/connector/stream/columns/types, assigned destination tables and loading mode in elt/loading.json",
        },
        "sources": sources,
        "targets": targets,
        "source_sha256": {
            source["name"]: hashlib.sha256(Path(source["options"]["path"]).read_bytes()).hexdigest()
            for source in sources
        },
        "reference_sha256": {
            target["name"]: hashlib.sha256(Path(target["expected"]).read_bytes()).hexdigest()
            for target in targets
        },
        "source_provenance": json.loads((assets / "source-provenance.json").read_text()),
        "reference_provenance": json.loads((assets / "reference-provenance.json").read_text()),
        "runtime": {
            "config_file": str(Path(runtime_config).resolve()),
            "benchmark_repo": str(Path(repo).resolve()),
        },
        "provenance": {
            "benchmark_repository": UPSTREAM,
            "benchmark_revision": ORIGINAL_REVISION,
            "execution_toolchain": "native_loading_v1; adapted from Airbyte/Terraform, not original platform execution",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "task.yaml"
    destination.write_text(yaml.safe_dump(data, sort_keys=False))
    return destination
