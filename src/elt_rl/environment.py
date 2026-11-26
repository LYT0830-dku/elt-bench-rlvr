"""Stateful ELT episode with grading against the submitted warehouse state."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from elt_rl.action_protocol import action_prefix, validate_action_fields
from elt_rl.artifacts import implementation_fingerprint, write_json
from elt_rl.errors import ActionError, InfrastructureError, IntegrityError
from elt_rl.project import Project
from elt_rl.rewards import REWARD_METRICS, RewardConfig
from elt_rl.source_data import source_rows
from elt_rl.timing import Timings
from elt_rl.warehouse import policy_error

REWARD_VERSION = "pure_cell_accuracy_v1"
DEFAULT_EPISODE_SECONDS: float | None = None
INSTRUCTIONS = """Build the supplied ELT task end to end using the native toolchain.
Call exactly one declared tool per response and wait for its actual result.
Read task/data_model.yaml, source schemas, original task configuration and task/native_loading.md.
This deployment adapts the execution tools: native loading replaces Airbyte Platform/Terraform,
while original source data, business specification, warehouse and official evaluator are preserved.
You must configure and execute extraction/loading yourself; no destination tables are preloaded.
inspect_task lists source capabilities, protocols, formats, columns and assigned destination.
Author elt/loading.json with format native_loading_v1, destination {"database":"assigned","schema":"airbyte_schema"},
and a connections list. Each connection needs name, source, connector, format, table, columns, sync_mode.
name is your connection name; source is a listed source ID; connector/format must match that source.
table selects the warehouse table. columns maps selected source column names to SQL types:
VARCHAR, BIGINT, DOUBLE, REAL, BOOLEAN, DATE, TIMESTAMP. Choose types appropriate for transformation.
sync_mode must be full_refresh_append or full_refresh_overwrite. No URLs, paths, credentials or code are accepted.
Run loading validate/plan/apply; apply registers your configuration without loading data.
Use sync with a returned native_connection.NAME, then wait_sync with its actual job_id to confirm loading.
Inspect warehouse tables with query. Repair configuration based on actual execution feedback.
Write dbt/dbt_project.yml (profile elt), model SQL and source YAML under dbt/models/.
dbt sources must use schema "{{ var('destination_schema') }}". Use ref/source/config/var declarative templates.
Only SELECT transformations and table/view materialization are allowed. Do not replace raw source tables.
Run dbt compile/run. You may revise and execute again using the compiler, SQL and loading feedback.
No shell, arbitrary Python, macros, hooks, plugins, external network destinations or host filesystem are available.
Submit only after your current loading files, syncs and current dbt files have successfully executed.
The submission is frozen and the trusted grader queries this episode's actual warehouse state.
Only submission earns execution-derived reward. Task inputs are immutable. Use edit_file for unique exact replacements.
"""


def uses_native_tools(task):
    path = getattr(task, "runtime", {}).get("config_file")
    return bool(path and json.loads(Path(path).read_text()).get("execution_backend") == "native")


def instructions_for(task):
    return INSTRUCTIONS


def initial_messages(observation):
    return action_prefix(
        observation["instructions"], loading=observation.get("execution_backend") == "native"
    ) + [
        {
            "role": "user",
            "content": json.dumps(
                {k: v for k, v in observation.items() if k != "instructions"}, default=str
            ),
        },
    ]


@dataclass(frozen=True)
class Step:
    observation: dict
    reward: float
    done: bool


class ELTEnvironment:
    def __init__(
        self,
        task,
        destination="duckdb",
        max_turns=80,
        max_seconds: float | None = DEFAULT_EPISODE_SECONDS,
        artifact_path=None,
        official_grader=None,
        *,
        allow_remote=False,
        reward_config=None,
        run_id=None,
        model_id=None,
        policy_checkpoint=None,
    ):
        try:
            valid_seconds = max_seconds is None or (
                not isinstance(max_seconds, bool)
                and isinstance(max_seconds, (int, float))
                and math.isfinite(max_seconds)
                and max_seconds > 0
            )
        except (TypeError, OverflowError):
            valid_seconds = False
        if max_turns < 1 or not valid_seconds:
            raise ValueError("Budgets must be positive")

        from elt_rl.task_spec import local_task_spec

        task = local_task_spec(task)
        self.task_spec = task
        self.reward_config = (
            dict(reward_config) if reward_config is not None else RewardConfig().configuration()
        )
        selected_reward = RewardConfig.from_dict(self.reward_config)
        self.run_id, self.model_id, self.policy_checkpoint = run_id, model_id, policy_checkpoint
        if self.task_spec:
            from elt_rl.evaluation import EVALUATORS
            from elt_rl.reward_registry import REWARDS

            if (
                destination not in task.evaluation.warehouse_backends
                or destination not in EVALUATORS.get(task.evaluation.provider).warehouse_backends
            ):
                raise ValueError("Task evaluation does not support this warehouse backend")
            REWARDS.get(selected_reward.type)
            task.verify()
        self.task, self.destination, self.allow_remote = task, destination, allow_remote
        self.implementation_sha256 = implementation_fingerprint()
        self.artifact_path = Path(artifact_path).resolve() if artifact_path else None
        self.root = (
            (self.artifact_path.parent / (self.artifact_path.stem + "-files"))
            if self.artifact_path
            else Path(tempfile.mkdtemp(prefix="elt-episode-"))
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        if not uses_native_tools(task):
            raise ValueError("Task runtime must configure the native execution backend")
        self.execution_backend = "native"
        self.project = Project(self.root / "project", task.project)
        self.rows = source_rows(task)
        self.runtime = None
        self.execution_result = None
        self.official_grader = official_grader
        self.timings = Timings()
        self.max_turns = max_turns
        self.max_seconds = max_seconds
        self.created_at = time.monotonic()
        self.started = None
        self.run_deadline: float | None = None
        self.dispatch_check = lambda: None
        self.turns = 0
        self.done = False
        self.closed = False
        self.trace = []
        self.initial_messages = []
        self.report = None
        self.termination = None
        self.invalid_json = None
        self.invalid_repeats = 0
        self.sampling = []
        self.resources = {}
        self.fingerprint = task.fingerprint()
        self.persist()

    def _execution_deadline(self):
        self.dispatch_check()
        deadlines = []
        if self.max_seconds is not None:
            deadlines.append(self.started + self.max_seconds)
        if self.run_deadline is not None:
            deadlines.append(self.run_deadline)
        return min(deadlines) if deadlines else None

    def _runtime(self):
        if self.started is None:
            self.started = time.monotonic()
        self.dispatch_check()
        if self.runtime is None:
            with self.timings.measure("environment_initialization"):
                from elt_rl.native_runtime import NativeRuntime

                self.runtime = NativeRuntime(
                    self.task,
                    self.project,
                    self.root / "exploration",
                    self.rows,
                    allow_remote=self.allow_remote,
                    deadline=self._execution_deadline(),
                    destination=self.destination,
                )
        return self.runtime

    def initial(self):
        if self.started is None:
            self.started = time.monotonic()
        observation = {
            "instructions": instructions_for(self.task),
            "execution_backend": self.execution_backend,
            "task": self.task.public(),
            "connection_facts": self._runtime().public(),
            "max_turns": self.max_turns,
            "max_seconds": self.max_seconds,
        }
        self.initial_messages = initial_messages(observation)
        self.persist()
        return observation

    def step(self, action, *, format_error=None):
        self.dispatch_check()
        if self.done or self.closed:
            raise RuntimeError("Episode is terminal or closed")
        if self.started is None:
            self.started = time.monotonic()
        self.turns += 1
        started = time.monotonic()
        parsed = None
        if self.max_seconds is not None and started - self.started > self.max_seconds:
            return self._finish({"error": "Episode time budget exceeded"}, 0)
        try:
            if format_error is not None:
                raise ActionError(format_error)
            if isinstance(action, str) and len(action) > 300_000:
                raise ActionError("Action exceeds size budget")
            parsed = json.loads(action) if isinstance(action, str) else dict(action)
            if not isinstance(parsed, dict):
                raise ActionError(
                    "Action must be a flat JSON object. Send exactly one action, "
                    'for example {"tool":"inspect_task"}, then wait for its result.'
                )
            tool, args = validate_action_fields(parsed)
            self.invalid_repeats = 0
            self.invalid_json = None
            runtime = self._runtime()
            with self.timings.measure("tool_" + tool):
                if tool == "inspect_task":
                    observation = {"task": self.task.public(), **runtime.public()}
                elif tool == "list_files":
                    observation = {"files": self.project.files()}
                elif tool == "read_file":
                    observation = {"content": self.project.read(args["path"])}
                elif tool == "write_file":
                    observation = self.project.write(args["path"], args["content"])
                elif tool == "edit_file":
                    observation = self.project.edit(args["path"], args["old"], args["new"])
                elif tool == "delete_file":
                    observation = self.project.delete(args["path"])
                elif tool == "loading":
                    observation = runtime.loading(**args)
                elif tool == "sync":
                    observation = runtime.sync(**args)
                elif tool == "wait_sync":
                    observation = runtime.wait_sync(**args)
                elif tool == "dbt":
                    observation = runtime.dbt_command(**args)
                elif tool == "query":
                    observation = runtime.warehouse.execute(args["sql"]).public()
                else:
                    observation = self._submit()
                    self.done = True
                    self.termination = "submitted"
        except json.JSONDecodeError as exc:
            self.invalid_repeats = self.invalid_repeats + 1 if self.invalid_json == action else 1
            self.invalid_json = action
            observation = {
                "error": "Invalid JSON: "
                + str(exc)
                + '. Send exactly one JSON object, for example {"tool":"inspect_task"}, '
                "then wait for its result."
            }
            if self.invalid_repeats >= 3:
                self.done = True
                self.termination = "Repeated identical invalid JSON"
        except (ActionError, ValueError, TypeError, KeyError) as exc:
            observation = {"error": str(exc)[:3000]}
        except Exception as exc:
            if policy_error(exc):
                observation = {"error": str(exc)[:3000]}
            else:
                self.done = True
                self.termination = "infrastructure_error"
                self.report = {
                    "failure_type": "infrastructure",
                    "training_eligible": False,
                    "training_reward": None,
                    "success": False,
                }
                self.trace.append(
                    {
                        "action": parsed,
                        "failure_type": "infrastructure",
                        "exception": type(exc).__name__,
                    }
                )
                self.persist()
                raise
        if self.turns >= self.max_turns and not self.done:
            self.done = True
            self.termination = "Maximum turns reached"
        if (
            self.max_seconds is not None
            and time.monotonic() - self.started > self.max_seconds
            and not self.done
        ):
            self.done = True
            self.termination = "Episode time budget exceeded"
        reward = (
            self.report["training_reward"] if self.report and self.termination == "submitted" else 0
        )
        if self.runtime and self.runtime.runner:
            observation = json.loads(
                self.runtime.runner.redact(json.dumps(observation, default=str))
            )
        self.trace.append(
            {
                "action": parsed if parsed is not None else action,
                "observation": observation,
                "seconds": time.monotonic() - started,
                "reward": reward,
                "done": self.done,
            }
        )
        self.persist()
        return Step(observation, reward, self.done)

    def _finish(self, observation, reward):
        self.done = True
        self.termination = observation.get("error", "terminated")
        self.persist()
        return Step(observation, reward, True)

    def _submit(self):
        if self.task.fingerprint() != self.fingerprint:
            raise IntegrityError("Trusted task inputs changed during episode")
        if implementation_fingerprint() != self.implementation_sha256:
            raise IntegrityError("Implementation changed during episode")
        runtime = self._runtime()
        if not runtime.valid():
            issues = runtime.submission_issues()
            raise ActionError("Submission is not ready: " + "; ".join(issues))
        bundle = self.root / "submission"
        self.project.freeze(bundle, runtime.recipe)
        bundle_hash = hashlib.sha256((bundle / "pipeline.json").read_bytes()).hexdigest()
        output = self.root / "submission-result"
        evidence = {
            "kind": "submission",
            "pipeline_valid": True,
            "execution_success": False,
            "execution": runtime.evidence(),
        }
        self.execution_result = evidence
        try:
            if self.task_spec:
                from elt_rl.evaluation import EVALUATORS
                from elt_rl.execution_scoring import score_submission

                with self.timings.measure("official_grading"):
                    outcome = EVALUATORS.get(self.task.evaluation.provider).evaluate(
                        runtime.warehouse, self.task, output, grader=self.official_grader
                    )
                official, success = outcome.metrics, outcome.success
                with self.timings.measure("training_reward"):
                    components, receipt = score_submission(
                        self.task,
                        output,
                        outcome,
                        self.reward_config,
                        loading_succeeded=True,
                    )
                reward = components["reward"]
                evidence["reward_result"] = receipt
                if outcome.grades:
                    evidence["grades"] = outcome.grades
            if (
                self.task.fingerprint() != self.fingerprint
                or implementation_fingerprint() != self.implementation_sha256
            ):
                raise IntegrityError("Trusted task or implementation changed during grading")
            evidence.update(execution_success=True, official=official)
        except Exception as exc:
            evidence["failure_type"] = "infrastructure"
            raise InfrastructureError("Submission grading failed; inspect the grader logs") from exc
        finally:
            write_json(output / "result.json", evidence)
        self.report = {
            **official,
            "reward_version": REWARD_VERSION,
            "training_reward": reward,
            "pipeline_valid": True,
            "execution_success": True,
            "success": success,
            "failure_type": None if success else "task_incomplete",
            "training_eligible": True,
            "bundle_sha256": bundle_hash,
        }
        if self.task_spec:
            self.report.update(
                reward_version=components["version"],
                reward_type=components["reward_type"],
                reward_result=receipt,
                **{key: components[key] for key in REWARD_METRICS if key in components},
            )
            if components["reward_type"] == "pure_cell":
                self.report["cell_results"] = components["cell_results"]
            # Terminal tool feedback is never a channel for private reference or
            # cell-level diagnostics. Scalar rewards travel via Step.reward.
            return {"submitted": True, "pipeline_valid": True, "execution_success": True}
        return self.report

    def artifact(self):
        return {
            "format": "elt_episode_v4",
            "mode": self.task.mode,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "policy_checkpoint": self.policy_checkpoint,
            "reward_configuration": self.reward_config if self.task_spec else None,
            "task": self.task.name,
            "fingerprint": self.fingerprint,
            "implementation_sha256": self.implementation_sha256,
            "destination": self.destination,
            "turns": self.turns,
            "done": self.done,
            "termination": self.termination,
            "report": self.report,
            "initial_messages": self.initial_messages,
            "trace": self.trace,
            "sampling": self.sampling,
            "timings": self.timings.events,
            "queue_seconds": (self.started - self.created_at) if self.started else None,
            "wall_clock_seconds": (time.monotonic() - self.started) if self.started else None,
            "runtime": self.runtime.evidence() if self.runtime else None,
            "execution_result": self.execution_result,
        }

    def persist(self):
        if self.artifact_path:
            write_json(self.artifact_path, self.artifact())

    def close(self):
        if self.closed:
            return
        try:
            if self.runtime:
                with self.timings.measure("cleanup"):
                    self.runtime.close()
            self.closed = True
        except InfrastructureError:
            self.done = True
            self.termination = "infrastructure_error"
            self.report = {
                "failure_type": "infrastructure",
                "training_eligible": False,
                "training_reward": None,
                "success": False,
            }
            raise
        finally:
            self.persist()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
