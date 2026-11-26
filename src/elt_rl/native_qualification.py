"""Host qualification of original source semantics; no model or ground-truth access.

Optional loading checks use a disposable namespace and never create target models.
This is engineering evidence, separate from a model-authored official rollout.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from datetime import date, datetime
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.errors import IntegrityError
from elt_rl.native_sources import NativeSourceServices
from elt_rl.source_data import source_rows


def expected_rows(source, decoded):
    if source.kind == "postgres":

        def convert(key, value):
            kind = source.options["initial_types"][key]
            if kind == "REAL":
                return struct.unpack("f", struct.pack("f", float(value)))[0]
            if kind in {"SERIAL", "INTEGER", "BIGINT"}:
                return int(value)
            if kind == "DOUBLE PRECISION":
                return float(value)
            if kind == "DATE":
                try:
                    return date.fromisoformat(value)
                except ValueError:
                    return datetime.strptime(value, source.options["initial_date_format"]).date()
            if kind == "TIMESTAMP":
                return datetime.fromisoformat(value)
            if kind == "BOOLEAN":
                return value.lower() in {"true", "t", "1"}
            return value

        return [{key: convert(key, value) for key, value in row.items()} for row in decoded]
    if source.kind == "rest":
        import numpy as np
        import pandas as pd

        records = pd.read_csv(source.options["path"]).replace({np.nan: None}).to_dict("records")
        return [{k.lower(): v for k, v in row.items()} for row in records]
    return decoded


def digest(rows):
    return hashlib.sha256(
        json.dumps(sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)).encode()
    ).hexdigest()


def warehouse_isolation(config, root):
    """Try actual cross-namespace reads with two simultaneously granted episode roles."""
    from snowflake.connector.errors import ProgrammingError

    from elt_rl.native_dbt_entrypoint import guard_connection
    from elt_rl.warehouse import SnowflakeWarehouse

    owned, events = [], []
    try:
        for _ in range(2):
            warehouse = SnowflakeWarehouse(config["snowflake"], config["snowflake_admin"])
            owned.append(warehouse)
            warehouse._query(
                f'CREATE TABLE {warehouse.quote(warehouse.schema)}."ISOLATION_SENTINEL" (VALUE INTEGER)',
                0,
            )
        for first, other in [(owned[0], owned[1]), (owned[1], owned[0])]:
            guard_connection(first.conn, first.role, root / "isolation-session-guards.jsonl")
            own_table = f'"{first.database}"."{first.schema}"."ISOLATION_SENTINEL"'
            first._query("SELECT * FROM " + own_table, 1)
            try:
                first._query(
                    f'SELECT * FROM "{first.database}"."{other.schema}"."ISOLATION_SENTINEL"', 1
                )
            except ProgrammingError as exc:
                if exc.errno != 2003 or exc.sqlstate not in {"02000", "42S02", "42501"}:
                    raise
                events.append(
                    {"cross_namespace_read": "denied", "sqlstate": exc.sqlstate, "errno": exc.errno}
                )
            else:
                raise IntegrityError("An episode role could read another episode namespace")
        return {"simultaneous_namespaces": 2, "secondary_roles": "NONE", "attempts": events}
    finally:
        failures = []
        for warehouse in reversed(owned):
            try:
                warehouse.close()
            except Exception as exc:
                failures.append(type(exc).__name__)
        if failures:
            raise IntegrityError("Owned isolation-test resource cleanup failed: " + str(failures))


def qualify_sources(task, output, *, warehouse=False):
    root = Path(output).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = json.loads(Path(task.runtime["config_file"]).read_text())
    if config.get("execution_backend") != "native" or os.getuid() == 0:
        raise IntegrityError("Qualification requires native backend and an ordinary user")
    decoded = source_rows(task)
    result = {
        "format": "native_qualification_v1",
        "uid": os.getuid(),
        "task": task.name,
        "task_fingerprint": task.fingerprint(),
        "sources": {},
        "model_used": False,
        "ground_truth_used": False,
        "warehouse_roundtrip": warehouse,
    }
    first = NativeSourceServices(task, decoded, root / "sources-a", config)
    second = None
    try:
        second = NativeSourceServices(task, decoded, root / "sources-b", config)
        ports_a = {s.server_port for s in first.servers}
        ports_b = {s.server_port for s in second.servers}
        if ports_a & ports_b or first.root == second.root or first.http_token == second.http_token:
            raise IntegrityError("Concurrent source capabilities overlap")
        for source in task.sources:
            capability = first.public()[source.name]
            actual = first.extract(
                source.name, capability["connector"], capability["format"], source.columns
            )
            expected = expected_rows(source, decoded[source.name])
            selected = [{c: row[c] for c in source.columns} for row in expected]
            if digest(actual) != digest(selected):
                raise IntegrityError("Original source semantics differ: " + source.name)
            result["sources"][source.name] = {
                "connector": capability["connector"],
                "format": capability["format"],
                "columns": source.columns,
                "row_count": len(actual),
                "records_sha256": digest(actual),
                "original_semantics_match": True,
            }
        result["concurrent_source_isolation"] = True
    finally:
        try:
            if second:
                second.close()
        finally:
            first.close()
    result["source_cleanup_confirmed"] = all(
        json.loads((root / name / "resources.json").read_text())["state"] == "closed"
        for name in ("sources-a", "sources-b")
    )
    if warehouse:
        from elt_rl.native_runtime import NativeRuntime
        from elt_rl.project import Project

        project = Project(root / "project", task.project)
        connections = [
            {
                "name": s.name,
                "source": s.name,
                "connector": result["sources"][s.name]["connector"],
                "format": result["sources"][s.name]["format"],
                "table": "qualification_" + s.name,
                "columns": {
                    c: {
                        "REAL": "REAL",
                        "DOUBLE PRECISION": "DOUBLE",
                        "TEXT": "VARCHAR",
                        "VARCHAR": "VARCHAR",
                        "SERIAL": "BIGINT",
                        "INTEGER": "BIGINT",
                        "BIGINT": "BIGINT",
                        "DATE": "DATE",
                        "TIMESTAMP": "TIMESTAMP",
                        "BOOLEAN": "BOOLEAN",
                    }[s.options["initial_types"][c]]
                    if s.kind == "postgres"
                    else "VARCHAR"
                    for c in s.columns
                },
                "sync_mode": "full_refresh_overwrite",
            }
            for s in task.sources
        ]
        project.write(
            "elt/loading.json",
            json.dumps(
                {
                    "format": "native_loading_v1",
                    "destination": {"database": "assigned", "schema": "airbyte_schema"},
                    "connections": connections,
                }
            ),
        )
        runtime = NativeRuntime(task, project, root / "runtime", decoded, allow_remote=True)
        try:
            runtime.loading("validate")
            runtime.loading("plan")
            runtime.loading("apply")
            result["loading"] = {}
            for connection in connections:
                address = "native_connection." + connection["name"]
                job = runtime.sync(address)
                feedback = runtime.wait_sync(job["job_id"])
                receipt = runtime.jobs[job["job_id"]]
                if (
                    receipt["state"] != "succeeded"
                    or not receipt["fidelity"]["overwrite_content_matches"]
                ):
                    raise IntegrityError("Warehouse contents differ from extracted source")
                result["loading"][connection["name"]] = {
                    "feedback": feedback,
                    "fidelity": receipt["fidelity"],
                }
            result["warehouse_namespace_isolated"] = True
        finally:
            runtime.close()
        result["warehouse_cleanup_confirmed"] = runtime.closed
        result["warehouse_permissions"] = warehouse_isolation(config, root)
    write_json(root / "qualification.json", result)
    return result
