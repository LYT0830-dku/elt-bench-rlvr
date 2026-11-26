"""Task-specific preparation adapters, separate from the trainer and loader."""

import json
from pathlib import Path

from elt_rl.registry import Registry
from elt_rl.task_adapters.cars import prepare_task
from elt_rl.task_adapters.declared import prepare_declared


def pack_definitions():
    return json.loads(Path(__file__).parents[1].joinpath("official_task_packs.json").read_text())


TASK_PREPARERS = Registry("official task-pack preparer")
TASK_PREPARERS.register("cars", prepare_task)
for _task_id in pack_definitions():
    if _task_id != "cars":
        TASK_PREPARERS.register(_task_id, prepare_declared)
