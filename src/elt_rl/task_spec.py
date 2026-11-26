"""Trusted task contracts: public execution inputs and private evaluation metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from elt_rl.errors import IntegrityError
from elt_rl.registry import Registry, UnsupportedCapability
from elt_rl.rewards import NumericReward, RewardField, RewardTarget, _parsed
from elt_rl.task import Task


@dataclass(frozen=True)
class AssetManifest:
    path: str
    sha256: str
    role: str
    rows: int | None = None


@dataclass(frozen=True)
class SourceSpec:
    id: str
    adapter: str
    columns: tuple[str, ...]
    configuration: dict
    required: bool = True

    @property
    def kind(self):
        return self.adapter

    @property
    def options(self):
        return self.configuration


@dataclass(frozen=True)
class EvaluationSpec:
    provider: str
    warehouse_backends: tuple[str, ...]
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    description: str
    official_revision: str | None
    execution_task: Task
    source_specs: tuple[SourceSpec, ...]
    reward_targets: tuple[RewardTarget, ...]
    assets: tuple[AssetManifest, ...]
    evaluation: EvaluationSpec
    required_tools: tuple[str, ...]
    alignment_verified: bool
    alignment_evidence: str
    resources: dict = field(default_factory=dict)
    specification_path: Path | None = None
    max_reference_rows: int = 2_000_000

    def __getattr__(self, name):
        return getattr(self.execution_task, name)

    def public(self):
        # A private type/alignment/reference contract never enters observations.
        return self.execution_task.public()

    def fingerprint(self):
        contract = {
            "id": self.id,
            "revision": self.official_revision,
            "targets": [asdict(target) for target in self.reward_targets],
            "assets": [asdict(asset) for asset in self.assets],
            "evaluation": asdict(self.evaluation),
            "tools": self.required_tools,
            "alignment_verified": self.alignment_verified,
            "alignment_evidence": self.alignment_evidence,
            "resources": self.resources,
            "specification_sha256": hashlib.sha256(self.specification_path.read_bytes()).hexdigest()
            if self.specification_path
            else None,
        }
        return json.loads(
            json.dumps(
                {**self.execution_task.fingerprint(), "framework_task_contract": contract},
                sort_keys=True,
            )
        )

    def verify(self, *, fine_grained=True):
        if self.id != self.execution_task.name:
            raise IntegrityError("TaskSpec ID and execution manifest disagree")
        if fine_grained and (not self.alignment_verified or not self.alignment_evidence):
            raise UnsupportedCapability(
                "Fine-grained reward requires an explicitly verified row-alignment rule"
            )
        if len(self.reward_targets) != len(self.execution_task.targets):
            raise IntegrityError("Reward target set differs from the execution contract")
        contracts = {target.name: target for target in self.execution_task.targets}
        for target in self.reward_targets:
            target.validate()
            original = contracts.get(target.name)
            if (
                original is None
                or set(field.name.casefold() for field in target.fields) != set(original.columns)
                or tuple(k.casefold() for k in target.keys) != original.keys
            ):
                raise IntegrityError("Target columns/keys differ from the execution contract")
            with original.expected.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                fields = {field.name.casefold(): field for field in target.fields}
                seen = set()
                for index, row in enumerate(reader):
                    if index >= self.max_reference_rows:
                        raise UnsupportedCapability(
                            "Reference exceeds configured host scoring row budget"
                        )
                    row = {key.casefold(): value for key, value in row.items()}
                    try:
                        parsed = {
                            name: _parsed(_csv_value(row[name], specification), specification)
                            for name, specification in fields.items()
                        }
                    except (KeyError, ValueError, ArithmeticError) as exc:
                        raise IntegrityError(
                            "Invalid trusted reference values: " + target.name
                        ) from exc
                    key = tuple(parsed[name.casefold()] for name in target.keys)
                    if any(value is None for value in key) or key in seen:
                        raise IntegrityError("Trusted row-alignment keys are NULL or duplicated")
                    seen.add(key)
                if not seen:
                    raise UnsupportedCapability(
                        "Empty reference requires a separately verified alignment contract"
                    )
        for asset in self.assets:
            if hashlib.sha256(Path(asset.path).read_bytes()).hexdigest() != asset.sha256:
                raise IntegrityError("Task asset digest changed: " + asset.path)
        return self


def _csv_value(value, specification):
    # CSV empty numeric/temporal fields are the documented NULL representation.
    # Empty strings remain literal strings; nullable string contracts use explicit
    # table-level values or a custom task adapter if their CSV format is ambiguous.
    return None if value == "" and specification.kind != "string" else value


class OfficialTaskLoader:
    """Load a complete, declared task pack; never infer missing assets or keys."""

    def load(self, path: str | Path) -> TaskSpec:
        path = Path(path).resolve(strict=True)
        raw = yaml.safe_load(path.read_text())
        allowed = {
            "format",
            "id",
            "manifest",
            "official_revision",
            "evaluation",
            "targets",
            "alignment",
            "required_tools",
            "resources",
            "max_reference_rows",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) - allowed
            or raw.get("format") != "elt_task_spec_v1"
        ):
            raise ValueError("Invalid/unknown TaskSpec fields")
        execution_task = Task.load((path.parent / raw["manifest"]).resolve())
        targets = []
        for target in raw["targets"]:
            if set(target) != {"name", "keys", "fields"}:
                raise ValueError("Target requires name, keys and fields")
            fields = []
            for definition in target["fields"]:
                definition = dict(definition)
                numeric = definition.pop("numeric", None)
                fields.append(
                    RewardField(
                        **definition,
                        numeric=NumericReward(**numeric) if numeric is not None else None,
                    )
                )
            targets.append(RewardTarget(target["name"], tuple(fields), tuple(target["keys"])))
        evaluation = raw["evaluation"]
        if set(evaluation) != {"provider", "warehouse_backends", "metrics"}:
            raise ValueError("Invalid evaluation contract")
        alignment = raw.get("alignment", {})
        if (
            set(alignment) - {"type", "verified", "evidence"}
            or alignment.get("type") != "primary_key"
        ):
            raise UnsupportedCapability(
                "Only explicitly verified primary/composite keys are currently supported"
            )
        if type(alignment.get("verified")) is not bool:
            raise ValueError("alignment.verified must be an explicit boolean")
        revision = raw.get("official_revision")
        actual_revision = execution_task.provenance.get("revision") or (
            yaml.safe_load(execution_task.manifest.read_text()).get("benchmark_lock") or {}
        ).get("revision")
        if execution_task.mode == "native" and revision != actual_revision:
            raise IntegrityError("TaskSpec official revision differs from the pinned task")
        assets = []
        for source in execution_task.sources:
            source_path = Path(source.options["path"])
            assets.append(
                AssetManifest(
                    str(source_path),
                    hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "source:" + source.name,
                )
            )
        for target in execution_task.targets:
            assets.append(
                AssetManifest(
                    str(target.expected),
                    hashlib.sha256(target.expected.read_bytes()).hexdigest(),
                    "reference:" + target.name,
                )
            )
        result = TaskSpec(
            raw["id"],
            execution_task.description,
            revision,
            execution_task,
            tuple(
                SourceSpec(
                    source.name, source.kind, source.columns, dict(source.options), source.required
                )
                for source in execution_task.sources
            ),
            tuple(targets),
            tuple(assets),
            EvaluationSpec(
                evaluation["provider"],
                tuple(evaluation["warehouse_backends"]),
                tuple(evaluation["metrics"]),
            ),
            tuple(raw.get("required_tools", ())),
            alignment["verified"],
            alignment.get("evidence", ""),
            dict(raw.get("resources", {})),
            path,
            raw.get("max_reference_rows", 2_000_000),
        )
        return result.verify()


class TaskRegistry(Registry[TaskSpec]):
    def __init__(self):
        super().__init__("task")

    def load(self, path):
        task = OfficialTaskLoader().load(path)
        return self.register(task.id, task)


def load_task_spec(path):
    return OfficialTaskLoader().load(path)


def local_task_spec(task):
    """Verified local integration contract; production tasks require an explicit TaskSpec."""
    if isinstance(task, TaskSpec):
        return task
    if task.mode != "local_test":
        raise UnsupportedCapability("Official tasks require an explicit verified TaskSpec")
    targets = tuple(
        RewardTarget(
            target.name,
            tuple(
                RewardField(
                    name,
                    kind="decimal" if name in target.numeric else "string",
                    role="identifier" if name in target.keys else "business",
                    comparator="relative"
                    if name in target.numeric and name not in target.keys
                    else "exact",
                )
                for name in target.columns
            ),
            target.keys,
        )
        for target in task.targets
    )
    assets = tuple(
        AssetManifest(
            str(target.expected),
            hashlib.sha256(target.expected.read_bytes()).hexdigest(),
            "reference:" + target.name,
        )
        for target in task.targets
    )
    result = TaskSpec(
        task.name,
        task.description,
        None,
        task,
        tuple(
            SourceSpec(
                source.name, source.kind, source.columns, dict(source.options), source.required
            )
            for source in task.sources
        ),
        targets,
        assets,
        EvaluationSpec("local_fixture", ("duckdb",), ("end_to_end_success", "column_accuracy")),
        ("native_loading_v1", "dbt"),
        True,
        "Explicit local fixture keys validated against its reference rows",
    )
    result.verify()
    return result


def load_task(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and raw.get("format") == "elt_task_spec_v1":
        return OfficialTaskLoader().load(path)
    return local_task_spec(Task.load(path))
