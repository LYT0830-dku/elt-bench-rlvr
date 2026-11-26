"""Structured native E/L and validated dbt, with no container or arbitrary-code API."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import yaml

from elt_rl.artifacts import write_json
from elt_rl.capabilities import dbt_policy
from elt_rl.errors import ActionError, InfrastructureError
from elt_rl.failures import check_dbt_result
from elt_rl.native_sources import NativeSourceServices
from elt_rl.runtime import Runtime
from elt_rl.task import identifier
from elt_rl.warehouse import policy_error

TYPES = {"VARCHAR", "BIGINT", "DOUBLE", "REAL", "BOOLEAN", "DATE", "TIMESTAMP"}
TYPE_ALIASES = {"STRING": "VARCHAR", "TEXT": "VARCHAR"}


def cast(value, kind):
    if value is None or (value == "" and kind != "VARCHAR"):
        return None
    try:
        if kind == "VARCHAR":
            return str(value)
        if kind == "BIGINT":
            from decimal import Decimal

            number = Decimal(str(value))
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("Not an integer")
            return int(number)
        if kind in {"DOUBLE", "REAL"}:
            import math

            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Nonfinite numeric source value")
            return number
        if kind == "BOOLEAN":
            if str(value).lower() not in {"true", "false", "1", "0"}:
                raise ValueError("Not a boolean")
            return str(value).lower() in {"true", "1"}
        if kind == "DATE":
            return date.fromisoformat(str(value))
        if kind == "TIMESTAMP":
            return datetime.fromisoformat(str(value))
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ActionError("Source value cannot be represented by the configured " + kind) from exc
    raise ActionError("Unsupported column type")


class NativeToolRunner:
    """Executes one fixed trusted dbt entry point after capability validation.

    The safety boundary is the restricted SQL/Jinja/YAML grammar, not subprocess
    or a virtual environment. No model-controlled executable, environment or args.
    """

    def __init__(self, root, config, secrets=(), deadline=None):
        self.root, self.deadline = root, deadline or float("inf")
        configured = config.get("dbt_executable") or os.environ.get("ELT_DBT_EXECUTABLE")
        if configured:
            self.binary = Path(configured).resolve()
        else:
            candidate = Path(sys.executable).parent / "dbt"
            local = Path(__file__).resolve().parents[2] / ".venv-dbt/bin/dbt"
            self.binary = candidate if candidate.is_file() else local
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise InfrastructureError(
                "Install the pinned dbt tool environment and configure dbt_executable"
            )
        self.secrets = [s for s in secrets if isinstance(s, str) and len(s) >= 4]
        self.events, self.active = [], set()

    def redact(self, value):
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]").replace(
                json.dumps(secret)[1:-1], "[REDACTED]"
            )
        return value

    def dbt(self, command, project, profiles, schema, role=""):
        if command not in {"compile", "run"}:
            raise ActionError("dbt command must be compile or run")
        args = [
            str(self.binary.parent / "python"),
            str(Path(__file__).with_name("native_dbt_entrypoint.py")),
            role,
            str(self.root / "dbt-snowflake-sessions.jsonl"),
            command,
            "--profiles-dir",
            str(profiles),
            "--project-dir",
            str(project),
            "--vars",
            json.dumps({"destination_schema": schema}),
            "--no-use-colors",
        ]
        home = self.root / "tool-home"
        home.mkdir(exist_ok=True, mode=0o700)
        env = {
            "PATH": str(self.binary.parent) + ":/usr/bin:/bin",
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
            "DO_NOT_TRACK": "1",
        }
        started = time.monotonic()
        remaining = self.deadline - started
        if remaining <= 0:
            from elt_rl.errors import EpisodeLimit

            raise EpisodeLimit("Episode time budget exceeded")
        process = subprocess.Popen(
            args,
            cwd=project,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.active.add(process)
        try:
            try:
                output, _ = process.communicate(timeout=min(600, remaining))
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                process.communicate(timeout=20)
                raise InfrastructureError(
                    "Native dbt execution timed out; no policy zero assigned"
                ) from exc
            result = {
                "command": ["dbt", command],
                "exit_code": process.returncode,
                "output": self.redact(output[-32000:]),
                "seconds": time.monotonic() - started,
                "execution_backend": "native",
                "timed_out": False,
            }
            self.events.append(result)
            return result
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            self.active.discard(process)

    def close(self):
        for process in list(self.active):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            self.active.discard(process)


class LockedWarehouse:
    def __init__(self, warehouse):
        self.warehouse, self.lock = warehouse, threading.RLock()

    def __getattr__(self, name):
        return getattr(self.warehouse, name)

    def execute(self, *args, **kwargs):
        with self.lock:
            return self.warehouse.execute(*args, **kwargs)

    def output(self, *args, **kwargs):
        with self.lock:
            return self.warehouse.output(*args, **kwargs)

    def close(self):
        with self.lock:
            if getattr(self, "backend", None) is not None:
                self.backend.cleanup()
            else:
                self.warehouse.close()


class NativeRuntime(Runtime):
    def __init__(
        self, task, project, root, rows, *, allow_remote=False, deadline=None, destination=None
    ):
        self.task, self.project, self.root, self.rows = task, project, Path(root).resolve(), rows
        self.root.mkdir(parents=True, mode=0o700)
        self.config = json.loads(Path(task.runtime["config_file"]).read_text())
        self.deadline = deadline or float("inf")
        self.events, self.resources, self.jobs, self.futures = [], {}, {}, {}
        self.latest_jobs, self.synced = {}, set()
        self.recipe = {
            "sync_connections": [],
            "dbt_command": "run",
            "loading_backend": "native_loading_v1",
        }
        self.applied = self.dbt_succeeded = self.closed = False
        self.applied_hashes = self.dbt_hashes = None
        self.sync_generation = 0
        self.table_generations, self.unreproducible_tables = {}, set()
        self.services = self.warehouse = self.runner = None
        self._state_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owned-elt-sync")
        try:
            from elt_rl.warehouse_backends import WAREHOUSES

            backend_type = WAREHOUSES.get(
                destination
                or self.config.get("warehouse_backend")
                or ("snowflake" if task.mode == "native" else "duckdb")
            )
            if backend_type.remote and not allow_remote:
                raise InfrastructureError(
                    "Native warehouse execution needs an authorized paid-run budget"
                )
            self.services = NativeSourceServices(
                task, rows, self.root / "sources", self.config, deadline
            )
            self.backend = backend_type.create(
                self.root, self.config, on_resources=lambda event: self._record("warehouse", event)
            )
            warehouse = self.backend.warehouse
            self.warehouse = LockedWarehouse(warehouse)
            self.warehouse.backend = self.backend
            secrets = []

            def collect(node, key=""):
                if isinstance(node, dict):
                    for k, v in node.items():
                        collect(v, k)
                elif isinstance(node, str) and any(
                    k in key.lower() for k in ("password", "secret", "token")
                ):
                    secrets.append(node)
                elif key == "private_key_file" and isinstance(node, str):
                    secrets.append(Path(node).read_text())

            collect(self.config)
            self.runner = NativeToolRunner(self.root, self.config, secrets, self.deadline)
            self.dbt, self.profiles = self.root / "dbt", self.root / "profiles"
            self.profiles.mkdir(mode=0o700)
            self._profile()
            self._record(
                "runtime",
                {"mode": task.mode, "execution_backend": "native", "schema": warehouse.schema},
            )
        except BaseException as exc:
            self.close()
            if isinstance(exc, Exception) and not isinstance(exc, InfrastructureError):
                raise InfrastructureError(
                    "Native host initialization failed; no policy reward assigned"
                ) from exc
            raise

    def _record(self, kind, value):
        with self._state_lock:
            self.events.append({"kind": kind, **value})
            write_json(self.root / "runtime.json", {"events": self.events, "jobs": self.jobs})

    def public(self):
        return {
            "mode": self.task.mode,
            "execution_backend": "native_loading_v1",
            "dialect": self.warehouse.dialect,
            "logical_schema": "airbyte_schema",
            "available_sources": self.services.public(),
            "destination": {"database": "assigned", "schema": "airbyte_schema"},
            "loading_config": "elt/loading.json",
            "resource_addresses": sorted(self.resources),
        }

    def evidence(self):
        return {**super().evidence(), "execution_backend": "native_loading_v1"}

    def _configuration(self):
        try:
            config = json.loads(self.project.read("elt/loading.json"))
        except json.JSONDecodeError as exc:
            raise ActionError("Invalid loading JSON: " + str(exc)) from exc
        if (
            not isinstance(config, dict)
            or set(config) != {"format", "destination", "connections"}
            or config["format"] != "native_loading_v1"
            or config["destination"] != {"database": "assigned", "schema": "airbyte_schema"}
        ):
            raise ActionError(
                "Use format native_loading_v1, assigned database and logical schema airbyte_schema"
            )
        entries = config["connections"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 24:
            raise ActionError("Declare between 1 and 24 loading connections")
        result = {}
        catalog = self.services.public()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "name",
                "source",
                "connector",
                "format",
                "table",
                "columns",
                "sync_mode",
            }:
                raise ActionError(
                    "Each connection needs name/source/connector/format/table/columns/sync_mode only"
                )
            try:
                name, table = identifier(entry["name"]), identifier(entry["table"])
            except ValueError as exc:
                raise ActionError(str(exc)) from exc
            source = entry["source"]
            if not isinstance(source, str) or source not in catalog:
                raise ActionError(
                    "Unknown source capability; inspect_task lists the available sources"
                )
            expected = catalog[source]
            if entry["connector"] != expected["connector"] or entry["format"] != expected["format"]:
                raise ActionError("Connector/format differs from this source's actual protocol")
            columns = entry["columns"]
            if (
                not isinstance(columns, dict)
                or not columns
                or set(columns) - set(expected["columns"])
            ):
                raise ActionError("Choose nonempty columns from the selected source schema")
            if any(not isinstance(t, str) for t in columns.values()):
                raise ActionError(
                    "Column types must be strings: VARCHAR/BIGINT/DOUBLE/REAL/BOOLEAN/DATE/TIMESTAMP"
                )
            normalized = {c: TYPE_ALIASES.get(t.upper(), t.upper()) for c, t in columns.items()}
            if set(normalized.values()) - TYPES:
                raise ActionError(
                    "Column types must be VARCHAR/BIGINT/DOUBLE/REAL/BOOLEAN/DATE/TIMESTAMP; "
                    "STRING and TEXT are accepted aliases for VARCHAR"
                )
            if entry["sync_mode"] not in {"full_refresh_append", "full_refresh_overwrite"}:
                raise ActionError("Select full_refresh_append or full_refresh_overwrite")
            address = "native_connection." + name
            if address in result or any(other["table"] == table for other in result.values()):
                raise ActionError("Connection names and destination tables must be unique")
            result[address] = {**entry, "table": table, "columns": normalized}
        return result

    def loading(self, command):
        if command not in {"validate", "plan", "apply"}:
            raise ActionError("Loading command must be validate/plan/apply")
        config = self._configuration()
        planned = [
            {
                "address": name,
                "source": entry["source"],
                "table": entry["table"],
                "columns": entry["columns"],
                "sync_mode": entry["sync_mode"],
            }
            for name, entry in config.items()
        ]
        if command == "apply":
            if any(not future.done() for future in self.futures.values()):
                raise ActionError("Wait for current sync jobs before reconfiguring loading")
            hashes = self.project.execution_hashes("elt")
            if hashes != self.applied_hashes:
                self.sync_generation += 1
                self.synced.clear()
                self.latest_jobs.clear()
                self.recipe["sync_connections"] = []
                self.dbt_succeeded = False
                self.dbt_hashes = None
            self.connections = config
            self.resources = {
                name: {
                    "id": name,
                    "type": "native_connection",
                    "streams": [entry["source"]],
                    "input": {"source": entry["source"]},
                }
                for name, entry in config.items()
            }
            self.applied_hashes, self.applied = hashes, True
            self._record("loading_applied", {"inputs": hashes, "connections": planned})
        return {"valid": True, "planned_connections": planned, "applied": command == "apply"}

    def sync(self, connection):
        if (
            not self.applied
            or self.applied_hashes != self.project.execution_hashes("elt")
            or connection not in self.resources
        ):
            raise ActionError(
                "Apply the current loading configuration and select one of its connections"
            )
        if connection in self.latest_jobs and not self.futures[self.latest_jobs[connection]].done():
            raise ActionError("This connection already has a running job")
        self.synced.discard(connection)
        self.dbt_succeeded, self.dbt_hashes = False, None
        job = "native-" + uuid.uuid4().hex
        self.jobs[job] = {
            "connection": connection,
            "state": "running",
            "generation": self.sync_generation,
        }
        self.latest_jobs[connection] = job
        entry = json.loads(json.dumps(self.connections[connection]))
        self.futures[job] = self.executor.submit(self._load, job, entry)
        self._record("sync_dispatched", {"job_id": job, "connection": connection})
        return self._job_feedback(job)

    def _job_feedback(self, job_id):
        job = dict(self.jobs[job_id])
        if job.get("failure_type") == "infrastructure":
            self.jobs[job_id]["infrastructure_observed"] = True
            raise InfrastructureError("Native sync infrastructure failure")
        return {
            "job_id": job_id,
            "state": job["state"],
            "feedback": job.get("feedback"),
            "fidelity": job.get("fidelity"),
        }

    @staticmethod
    def _source_cell(value, kind, entry, column):
        try:
            return cast(value, kind)
        except ActionError as exc:
            raise ActionError(
                f"Connection {entry['name']}, source {entry['source']}, column {column}: "
                f"{exc}; source value {repr(value)[:160]}. "
                "Select a compatible loading type; perform data cleaning in dbt."
            ) from exc

    def _load(self, job, entry):
        started = time.monotonic()
        try:
            columns, types = tuple(entry["columns"]), tuple(entry["columns"].values())
            records = self.services.extract(
                entry["source"], entry["connector"], entry["format"], columns
            )
            values = [
                tuple(
                    self._source_cell(row[c], t, entry, c)
                    for c, t in zip(columns, types, strict=True)
                )
                for row in records
            ]
            with self.warehouse.lock:
                if (
                    entry["sync_mode"] == "full_refresh_append"
                    and self.table_generations.get(entry["table"]) != self.sync_generation
                ):
                    if self.backend.table_exists(entry["table"]):
                        self.unreproducible_tables.add(entry["table"])
                self._write_table(entry["table"], columns, types, values, entry["sync_mode"])
                if entry["sync_mode"] == "full_refresh_overwrite":
                    self.unreproducible_tables.discard(entry["table"])
                self.table_generations[entry["table"]] = self.sync_generation
                actual = self.warehouse.output(entry["table"])
                physical_types = self.backend.physical_types(entry["table"])
            expected = sorted(json.dumps(row, default=str) for row in values)
            actual_values = [
                tuple(
                    cast(row[actual.columns.index(c)], t)
                    for c, t in zip(columns, types, strict=True)
                )
                for row in actual.rows
            ]
            obtained = sorted(json.dumps(row, default=str) for row in actual_values)
            identity = {
                "columns": columns,
                "types": types,
                "warehouse_column_types": physical_types,
                "source_rows": len(values),
                "warehouse_rows": len(actual.rows),
                "source_values_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                "warehouse_values_sha256": hashlib.sha256(
                    json.dumps(obtained).encode()
                ).hexdigest(),
                "overwrite_content_matches": expected == obtained
                if entry["sync_mode"] == "full_refresh_overwrite"
                else None,
            }
            self.jobs[job].update(
                state="succeeded",
                rows_loaded=len(values),
                fidelity=identity,
                seconds=time.monotonic() - started,
            )
            self.recipe["sync_connections"] = [
                node["connection"]
                for node in self.jobs.values()
                if node["generation"] == self.sync_generation and node["state"] == "succeeded"
            ]
        except ActionError as exc:
            self.unreproducible_tables.add(entry["table"])
            self.jobs[job].update(state="failed", failure_type="policy", feedback=str(exc)[:1500])
        except Exception as exc:
            self.unreproducible_tables.add(entry["table"])
            if policy_error(exc):
                self.jobs[job].update(
                    state="failed",
                    failure_type="policy",
                    feedback=self.runner.redact(str(exc))[:1500],
                )
            else:
                self.jobs[job].update(
                    state="failed", failure_type="infrastructure", exception=type(exc).__name__
                )
                raise InfrastructureError(
                    "Native extraction/loading infrastructure failed"
                ) from exc
        finally:
            self._record("sync_finished", {"job_id": job, **self.jobs[job]})

    def _write_table(self, table, columns, types, values, mode):
        return self.backend.load(table, columns, types, values, mode)

    def wait_sync(self, job_id):
        if job_id not in self.jobs:
            raise ActionError("Unknown job; sync jobs are episode-owned")
        job = self.jobs[job_id]
        if (
            job["generation"] != self.sync_generation
            or self.latest_jobs.get(job["connection"]) != job_id
        ):
            raise ActionError("This job belongs to an obsolete connection execution")
        try:
            self.futures[job_id].result(
                timeout=max(0.1, min(300, self.deadline - time.monotonic()))
            )
        except InfrastructureError:
            job["infrastructure_observed"] = True
            raise
        if job["state"] == "succeeded":
            self.synced.add(job["connection"])
        elif job.get("failure_type") == "infrastructure":
            job["infrastructure_observed"] = True
            raise InfrastructureError("Native sync infrastructure failure")
        self._record("sync_result", {"job_id": job_id, "state": job["state"]})
        return self._job_feedback(job_id)

    def _profile(self):
        target = self.backend.profile()
        path = self.profiles / "profiles.yml"
        path.write_text(
            yaml.safe_dump({"elt": {"target": "episode", "outputs": {"episode": target}}})
        )
        path.chmod(0o600)

    def dbt_command(self, command):
        if command not in {"compile", "run"}:
            raise ActionError("dbt command must be compile or run")
        dbt_policy(self.project.root / "dbt", self.warehouse.dialect)
        if any(
            p.stem.lower()
            in (
                {s.name for s in self.task.sources}
                | {entry["table"] for entry in getattr(self, "connections", {}).values()}
            )
            for p in (self.project.root / "dbt/models").rglob("*.sql")
        ):
            raise ActionError("dbt models cannot replace synchronized source tables")
        if any(not f.done() for f in self.futures.values()):
            raise ActionError("Wait for sync jobs before executing transformations")
        self.dbt_succeeded, self.dbt_hashes = False, None
        if command == "run":
            keep = {self.connections[address]["table"] for address in self.synced}
            if keep & self.unreproducible_tables:
                raise ActionError(
                    "Append contains data from an older loading configuration or failed sync; "
                    "use full_refresh_overwrite and sync again before dbt run"
                )
            with self.warehouse.lock:
                removed = self.warehouse.prune_relations(keep)
            self._record(
                "warehouse_reconciled", {"kept_source_tables": sorted(keep), "removed": removed}
            )
        if self.dbt.exists():
            shutil.rmtree(self.dbt)
        shutil.copytree(self.project.root / "dbt", self.dbt)
        with self.warehouse.lock:
            result = self.backend.transformation(self.runner, command, self.dbt, self.profiles)
        check_dbt_result(result)
        self.dbt_succeeded = command == "run" and result["exit_code"] == 0
        self.dbt_hashes = self.project.execution_hashes("dbt") if self.dbt_succeeded else None
        return result

    def valid(self):
        return super().valid() and not any(
            self.connections[address]["table"] in self.unreproducible_tables
            for address in self.synced
        )

    def submission_issues(self):
        """Explain operational prerequisites without inspecting target answers."""
        issues = []
        if not self.applied:
            issues.append("Loading configuration has not been applied")
        elif self.applied_hashes != self.project.execution_hashes("elt"):
            issues.append("Loading configuration changed; apply it again")
        covered = {
            source
            for address in self.synced
            for source in self.resources.get(address, {}).get("streams", [])
        }
        missing = {s.name for s in self.task.sources if s.required} - covered
        if missing:
            issues.append("Required source syncs not confirmed: " + ", ".join(sorted(missing)))
        for address, job_id in sorted(self.latest_jobs.items()):
            job = self.jobs[job_id]
            if job["state"] == "failed":
                issues.append(f"{address} failed: {job.get('feedback', 'sync failed')}")
            elif address not in self.synced:
                issues.append(f"Call wait_sync for {job_id} ({address}, {job['state']})")
        if self.synced != set(self.recipe["sync_connections"]):
            issues.append("Confirm the latest successful sync jobs with wait_sync")
        if not self.dbt_succeeded:
            issues.append("No successful dbt run for the current loading state")
        elif self.dbt_hashes != self.project.execution_hashes("dbt"):
            issues.append("dbt project changed; run dbt again")
        if any(
            self.connections[address]["table"] in self.unreproducible_tables
            for address in self.synced
        ):
            issues.append("Replace stale append data with full_refresh_overwrite and sync again")
        return issues

    def close(self):
        if self.closed:
            return
        self.executor.shutdown(wait=True, cancel_futures=True)
        super().close()
        if any(
            job.get("failure_type") == "infrastructure" and not job.get("infrastructure_observed")
            for job in self.jobs.values()
        ):
            raise InfrastructureError(
                "Native sync infrastructure failed, including jobs not inspected with wait_sync"
            )
