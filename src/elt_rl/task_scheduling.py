"""Task groups share a policy, never a reward baseline across unrelated tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from elt_rl.registry import UnsupportedCapability
from elt_rl.task_spec import TaskRegistry


@dataclass(frozen=True)
class TaskGroup:
    task_id: str
    specification: str


class TaskSchedule:
    def __init__(self, groups, evaluation_task=None):
        self.groups = tuple(
            TaskGroup(**group) if isinstance(group, dict) else group for group in groups
        )
        if not self.groups or len({group.task_id for group in self.groups}) != len(self.groups):
            raise ValueError("Task schedule requires unique declared tasks")
        self.evaluation_task = evaluation_task or self.groups[0].task_id
        if self.evaluation_task not in {group.task_id for group in self.groups}:
            raise UnsupportedCapability(
                "Evaluation task must be explicitly loaded in the task registry"
            )

    def select(self, label):
        match = re.fullmatch(r"train-batch-(\d+)", label)
        if match:
            number = int(match[1])
            if number < 1:
                raise ValueError("Training batches are numbered from 1")
            return self.groups[(number - 1) % len(self.groups)]
        if label.startswith(("eval", "final")):
            return next(group for group in self.groups if group.task_id == self.evaluation_task)
        raise ValueError("Unrecognized cohort label for a task schedule")

    def load(self):
        registry = TaskRegistry()
        for group in self.groups:
            task = registry.load(group.specification)
            if task.id != group.task_id:
                raise ValueError("Task schedule ID and specification disagree")
        return TaskCollection(registry, self)


class TaskCollection:
    def __init__(self, registry, schedule):
        self.registry, self.schedule = registry, schedule
        self.primary = registry.get(schedule.groups[0].task_id)

    def __getattr__(self, name):
        return getattr(self.primary, name)

    def fingerprint(self):
        return {
            "tasks": {
                name: self.registry.get(name).fingerprint() for name in self.registry.names()
            },
            "task_schedule": [vars(group) for group in self.schedule.groups],
            "evaluation_task": self.schedule.evaluation_task,
        }

    def public(self):
        if len(self.schedule.groups) == 1:
            return self.primary.public()
        return {
            "tasks": [self.registry.get(name).public() for name in self.registry.names()],
            "group_schedule": "round_robin_same_task_within_group",
            "evaluation_task": self.schedule.evaluation_task,
        }


def expected_task_fingerprint(inputs, label):
    task = inputs["task"]
    if "task_schedule" not in task:
        return task
    selected = TaskSchedule(task["task_schedule"], task["evaluation_task"]).select(label)
    return task["tasks"][selected.task_id]
