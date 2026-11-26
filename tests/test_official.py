"""Original-task provenance checks using an explicitly synthetic local Git fixture."""

import json
import subprocess

import pytest
import yaml

from elt_rl import benchmark
from elt_rl.errors import IntegrityError
from elt_rl.fetch import fetch_assets


def fixture_repository(tmp_path, monkeypatch):
    files = {
        "elt-bench/tiny/config.yaml": "{}",
        "elt-bench/tiny/data_model.yaml": yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "result",
                        "columns": [{"name": "id", "description": "item identifier"}],
                    }
                ]
            }
        ),
        "elt-bench/tiny/schemas/items.csv": "column_name,column_description\nid,identifier\n",
        "agents/SWE-agent/config/elt.yaml": "fixture prompt",
        "setup/main.tf": "terraform {}",
        "setup/check_job_status.py": "pass",
        "evaluation/eva.py": "pass",
        "evaluation/eva_stage1.py": "pass",
        "evaluation/eva_stage2.py": "pass",
        "evaluation/table.json": json.dumps({"tiny": {"items": 1}}),
        "setup/elt_bench.yaml": "type: DeclarativeSource",
        "elt-docker/rest_api/app.py": "pass",
        "setup/elt_setup.sh": "# fixture",
        "evaluation/tiny/result.sql": "SELECT 1 AS id",
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "synthetic metadata fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(benchmark, "ORIGINAL_REVISION", revision)
    return tmp_path, revision


def test_selected_git_blobs_are_locked(tmp_path, monkeypatch):
    repo, revision = fixture_repository(tmp_path, monkeypatch)
    files, lock = benchmark.task_material(repo, "tiny")
    assert lock["revision"] == revision and lock["source_counts"] == {"items": 1}
    assert "elt-bench/tiny/schemas/items.csv" in lock["files_sha256"]
    assert files["setup/main.tf"] == b"terraform {}"


@pytest.mark.parametrize("change", ["changed_model", "deleted_schema", "changed_grader"])
def test_modified_original_material_fails_before_download(tmp_path, monkeypatch, change):
    repo, _ = fixture_repository(tmp_path, monkeypatch)
    path = {
        "changed_model": "elt-bench/tiny/data_model.yaml",
        "deleted_schema": "elt-bench/tiny/schemas/items.csv",
        "changed_grader": "evaluation/eva_stage2.py",
    }[change]
    if change == "deleted_schema":
        (repo / path).unlink()
    else:
        (repo / path).write_text("modified")
    monkeypatch.setattr(
        "elt_rl.fetch.public_archive",
        lambda *args, **kwargs: pytest.fail("HTTP called before provenance verification"),
    )
    with pytest.raises(IntegrityError, match="Modified original benchmark file"):
        fetch_assets(repo, "tiny", tmp_path / "out")


def test_unpinned_revision_is_rejected_before_material_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark.subprocess, "check_output", lambda *args, **kwargs: "different-revision\n"
    )
    monkeypatch.setattr(
        "elt_rl.fetch.public_archive", lambda *args, **kwargs: pytest.fail("Unexpected HTTP")
    )
    with pytest.raises(IntegrityError, match="requires"):
        fetch_assets(tmp_path, "cars", tmp_path / "output")
