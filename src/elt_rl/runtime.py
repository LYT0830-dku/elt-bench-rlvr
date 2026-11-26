"""Shared submission consistency and owned-resource cleanup for native execution."""

import json

from elt_rl.errors import InfrastructureError


class Runtime:
    def valid(self):
        required = {s.name for s in self.task.sources if s.required}
        covered = set()
        for address in self.synced:
            resource = self.resources.get(address, {})
            if self.task.mode == "local_test":
                covered.add((resource.get("input") or {}).get("source"))
            else:
                covered.update(resource.get("streams", []))
        return (
            self.applied
            and self.applied_hashes == self.project.execution_hashes("elt")
            and self.dbt_succeeded
            and self.dbt_hashes == self.project.execution_hashes("dbt")
            and bool(self.synced)
            and self.synced == set(self.recipe["sync_connections"])
            and required <= covered
            and all(
                self.jobs[self.latest_jobs[address]]["state"] == "succeeded"
                for address in self.synced
            )
        )

    def evidence(self):
        payload = {
            "mode": self.task.mode,
            "resources": {k: {"id": v["id"], "type": v["type"]} for k, v in self.resources.items()},
            "jobs": self.jobs,
            "events": self.events,
            "tools": self.runner.events if self.runner else [],
            "recipe": self.recipe,
            "execution_inputs": {
                "loading": self.applied_hashes,
                "dbt": self.dbt_hashes,
                "sync_generation": self.sync_generation,
                "latest_jobs": self.latest_jobs,
            },
            "pipeline_valid": self.valid(),
        }
        if self.runner:
            # Job diagnostics can contain connector messages, not just tool stdout.
            return json.loads(self.runner.redact(json.dumps(payload, default=str)))
        return payload

    def close(self):
        if self.closed:
            return
        failures = []
        quiesced = True
        if self.runner:
            try:
                self.runner.close()
            except Exception as exc:
                quiesced = False
                failures.append(str(exc))
        if self.warehouse and quiesced:
            try:
                self.warehouse.close()
            except Exception as exc:
                failures.append(str(exc))
        if self.services and quiesced:
            try:
                self.services.close()
            except Exception as exc:
                failures.append(str(exc))
        self.closed = not failures
        self._record(
            "cleanup", {"status": "failed" if failures else "closed", "failures": failures}
        )
        if failures:
            raise InfrastructureError("Runtime cleanup needs reconciliation: " + str(failures))
