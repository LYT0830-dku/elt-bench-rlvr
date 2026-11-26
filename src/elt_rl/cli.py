"""Unified entry point for native ELT execution, training and evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from elt_rl import example_dir


def experiment(path, allow_remote=False):
    from elt_rl.framework_config import lower_configuration

    path = Path(path).resolve()
    data = (
        json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    )
    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a mapping")
    return lower_configuration(data, path, allow_remote)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch-assets", "prepare-task"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--name", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare-task":
            command.add_argument("--assets", type=Path, required=True)
            command.add_argument("--runtime-config", type=Path, required=True)
    command = commands.add_parser("demo")
    command.add_argument("--task", type=Path, default=example_dir() / "task.yaml")
    command.add_argument("--project", type=Path, default=example_dir() / "pipeline.json")
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("run-project")
    for field in ("task", "project", "output"):
        command.add_argument("--" + field, type=Path, required=True)
    command.add_argument("--allow-remote", action="store_true")
    command = commands.add_parser("inspect-task")
    command.add_argument("--task", type=Path, required=True)
    for name in ("train", "evaluate", "inspect-experiment"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "inspect-experiment":
            command.add_argument("--allow-remote", action="store_true")
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path)
    commands.add_parser("configure-tinker")
    command = commands.add_parser("prepare-snowflake")
    command.add_argument("--account", required=True)
    command.add_argument("--manager", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-assets":
            from elt_rl.fetch import fetch_assets

            print(fetch_assets(args.repo, args.name, args.output))
        elif args.command == "prepare-task":
            from elt_rl.benchmark import prepare_task

            print(prepare_task(args.repo, args.name, args.assets, args.runtime_config, args.output))
        elif args.command == "inspect-task":
            from elt_rl.task_spec import load_task

            task = load_task(args.task)
            print(json.dumps(task.public(), indent=2))
        elif args.command == "inspect-experiment":
            config = experiment(args.config)
            config.validate()
            process = (
                (1 + config.batches // config.evaluation_interval) * config.eval_episodes
                if config.evaluation_enabled
                else 0
            )
            final = 2 * config.final_eval_episodes if config.evaluation_enabled else 0
            print(
                json.dumps(
                    {
                        "run_id": config.run_id,
                        "model": config.model,
                        "training_batches": config.batches,
                        "training_rollouts": config.batches * config.group_size,
                        "process_evaluation_rollouts": process,
                        "final_evaluation_rollouts": final,
                        "total_rollouts": config.batches * config.group_size + process + final,
                    },
                    indent=2,
                )
            )
        elif args.command == "configure-tinker":
            from elt_rl.local_auth import configure_interactively

            print("Saved private credential:", configure_interactively())
        elif args.command == "prepare-snowflake":
            from elt_rl.snowflake_setup import prepare, prepare_manager

            paths = (
                prepare_manager(
                    {"account": args.account, "database": "ELT_RL_DB", "warehouse": "ELT_RL_WH"},
                    Path.cwd(),
                )
                if args.manager
                else prepare(args.account)
            )
            print("Generated warehouse setup:", paths["bootstrap"])
        elif args.command in {"train", "evaluate"}:
            from elt_rl.local_auth import load_local_key
            from elt_rl.train import evaluate_model, train

            config = experiment(args.config, args.allow_remote)
            load_local_key()
            result = (
                train(config)
                if args.command == "train"
                else evaluate_model(config, str(args.checkpoint) if args.checkpoint else None)
            )
            asyncio.run(result)
        elif args.command in {"demo", "run-project"}:
            from elt_rl.pipeline import load_pipeline, run_pipeline
            from elt_rl.task_spec import load_task

            task = load_task(args.task)
            result = run_pipeline(
                task,
                load_pipeline(args.project),
                destination="snowflake" if task.mode == "native" else "duckdb",
                artifact_path=args.output,
                allow_remote=getattr(args, "allow_remote", False),
            )
            print(json.dumps(result.get("report"), indent=2))
            return 0 if (result.get("report") or {}).get("success") else 1
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
