import pytest

from elt_rl import example_dir
from elt_rl.task import Task


@pytest.fixture
def task_path():
    return example_dir() / "task.yaml"


@pytest.fixture
def pipeline_path():
    return example_dir() / "pipeline.json"


@pytest.fixture
def task(task_path):
    return Task.load(task_path)
