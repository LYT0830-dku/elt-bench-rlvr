"""Model write capabilities and immutable submission integrity."""

import hashlib
import json

import pytest

from elt_rl.environment import ELTEnvironment
from elt_rl.errors import ActionError
from elt_rl.pipeline import load_pipeline
from elt_rl.project import Project


@pytest.mark.parametrize(
    "path",
    [
        "../outside.tf",
        "elt/unused.tf",
        "task/spec.txt",
        "dbt/macros/escape.sql",
        "dbt/models/out;select.sql",
        "/tmp/x.tf",
    ],
)
def test_private_or_executable_paths_are_rejected(tmp_path, path):
    project = Project(tmp_path / "project", {"task/spec.txt": "task"})
    with pytest.raises(ActionError):
        project.write(path, "content")


def test_starter_is_editable_but_submission_freezes_exact_files(tmp_path, pipeline_path):
    project = Project(
        tmp_path / "project", {"task/spec.txt": "immutable task", "elt/loading.json": "starter"}
    )
    pipeline = load_pipeline(pipeline_path)
    for path, content in pipeline["files"].items():
        project.write(path, content)
    bundle = tmp_path / "bundle"
    recipe = {
        "sync_connections": pipeline["sync_connections"],
        "dbt_command": "run",
        "loading_backend": pipeline["loading_backend"],
    }
    manifest = project.freeze(bundle, recipe)
    assert load_pipeline(bundle / "pipeline.json") == pipeline
    assert json.loads((bundle / "pipeline.json").read_text()) == manifest
    for name, content in pipeline["files"].items():
        assert (bundle / name).read_text() == content
        assert manifest["files"][name] == hashlib.sha256(content.encode()).hexdigest()


def test_edit_file_preserves_surrounding_text_and_returns_hash(task, tmp_path, monkeypatch):
    monkeypatch.setattr(ELTEnvironment, "_runtime", lambda self: None)
    with ELTEnvironment(task, artifact_path=tmp_path / "episode.json") as env:
        before = "first = 1\nsecond = 2\nthird = 3\n"
        env.project.write("elt/loading.json", before)
        result = env.step(
            {
                "tool": "edit_file",
                "path": "elt/loading.json",
                "old": "second = 2",
                "new": "second = 4",
            }
        )
        after = "first = 1\nsecond = 4\nthird = 3\n"
        assert env.project.read("elt/loading.json") == after
        assert result.observation == {
            "written": "elt/loading.json",
            "sha256": hashlib.sha256(after.encode()).hexdigest(),
        }
        assert not result.done


def test_edit_file_rejects_invalid_matches_and_reuses_write_guards(tmp_path, monkeypatch):
    project = Project(
        tmp_path / "project", {"task/spec.txt": "immutable", "elt/loading.json": "aaaa"}
    )
    for old, new in [
        ("", "x"),
        (None, "x"),
        ("a", None),
        ("missing", "x"),
        ("a", "x"),
        ("aaa", "x"),
    ]:
        with pytest.raises(ActionError):
            project.edit("elt/loading.json", old, new)
        assert project.read("elt/loading.json") == "aaaa"
    for path in ["task/spec.txt", "../outside.tf", "dbt/macros/escape.sql"]:
        with pytest.raises(ActionError):
            project.edit(path, "immutable", "changed")
    with pytest.raises(ActionError, match="File does not exist"):
        project.edit("elt/missing.json", "old", "new")
    monkeypatch.setattr("elt_rl.project.MAX_FILE_BYTES", 8)
    with pytest.raises(ActionError, match="write budget"):
        project.edit("elt/loading.json", "aaaa", "x" * 9)
    monkeypatch.setattr("elt_rl.project.MAX_PROJECT_BYTES", 13)
    with pytest.raises(ActionError, match="Project exceeds"):
        project.edit("elt/loading.json", "aaaa", "x" * 8)
    project.submitted = True
    with pytest.raises(ActionError, match="frozen"):
        project.edit("elt/loading.json", "aaaa", "changed")
    assert project.read("elt/loading.json") == "aaaa"
    assert (project.root / "task/spec.txt").read_text() == "immutable"
